import albumentations as A

IMG_SIZE = 128

Augmentation_Train1 = A.Compose([
    A.Rotate(limit=20, interpolation=1, border_mode=1, value=None, mask_value=None, always_apply=False, p=0.25),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=0, border_mode=0, always_apply=False, p=0.25),
    A.RandomBrightnessContrast(always_apply=False, p=0.25),
    ], p=0.25)

Augmentation_Test1 = A.Compose([
    A.OneOf([
        A.MultiplicativeNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.3),
        ], p=1.0),

    ], p=1)


Augmentation_N = A.Compose([A.GaussNoise(mean=0, always_apply=False, p=0.1)], p=0.1)

