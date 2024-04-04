import logging
from typing import Optional

import numpy as np
import torch
from mivolo.data.misc import prepare_classification_images
from mivolo.model.create_timm_model import create_model
from mivolo.structures import PersonAndFaceCrops, PersonAndFaceResult
from timm.data import resolve_data_config

_logger = logging.getLogger("MiVOLO")
has_compile = hasattr(torch, "compile")


class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2
        self.input_size = 224

    def load_from_ckpt(self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = True if "patch_embed.conv1.0.weight" in state["state_dict"] else False

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError(
                "You can not disable faces and persons together. "
                "Set --with-persons if you want to run with --disable-faces"
            )
        self.input_size = state["state_dict"]["pos_embed"].shape[1] * 16
        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model


class MiVOLO:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
    ):
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        self.meta: Meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)
        if self.verbose:
            _logger.info(f"Model meta:\n{str(self.meta)}")

        model_name = f"mivolo_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        self.param_count = sum([m.numel() for m in self.model.parameters()])
        _logger.info(f"Model {model_name} created, param count: {self.param_count}")

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )

        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        self.model = self.model.to(self.device)

        if torchcompile:
            assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        self.model.eval()
        if self.half:
            self.model = self.model.half()

    def inference(self, model_input: torch.tensor) -> torch.tensor:

        with torch.no_grad():
            if self.half:
                model_input = model_input.half()
            output = self.model(model_input)
        return output

    def predict(self, image: np.ndarray, faces_list ):

        faces_input = self.prepare_crops(image, faces_list)

        
        model_input = faces_input
        output = self.inference(model_input)

        # write gender and age results into detected_bboxes
        return self.fill_in_results(output)
         
    def fill_in_results(self, output):
        
        age_output = output[:, 2]
        gender_output = output[:, :2].softmax(-1)
        gender_probs, gender_indx = gender_output.topk(1)


        # per face
        self.ages = []
        self.genders = []
        for index in range(output.shape[0]):
          

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)
            self.ages.append(age)

            if gender_probs is not None:
                gender = "male" if gender_indx[index].item() == 0 else "female"
                self.genders.append(gender)
                gender_score = gender_probs[index].item()
        return self.ages,self.genders
    def prepare_crops(self, image: np.ndarray, detected_bboxes):
        '''
        detected_bboxes: [[x1,y1,x2,y2]]
        
        '''
        x1,y1,x2,y2 = detected_bboxes[0]
        face_crop = image[y1:y2,x1:x2,:]
        faces_input = prepare_classification_images(
            [face_crop], self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        return faces_input

if __name__ == "__main__":
    model = MiVOLO("../pretrained/checkpoint-377.pth.tar", half=True, device="cuda:0")
