## Log 
### Monday 11th of May
* Data preprocessing and investigation
* Clean up the data labels.
* Make train, test and val splits for pretraining and transfer learning.

### Tuesday 12th of May
* Trimming pre-processing script
* Dataset definitions and tests
* Cleaning image data
* Add pad to square and resizing operations to homogenize the images

### Wednesday 13th of May
* add imagenet rgb norm
* add losses
* add network definitions
* add model definition
* first pretraining 

### Thursday 14th of May
* add confusion matrix 
* add f-scores metric

### Friday 15th of May
* add focal loss

### Saturday 16th of May
* add Flip Left Right Augmentation
* make inference script for test

### Sunday 17th of May
* Running pretrain experiments on Google Colab
* Experiences: works well for a limited set of the data but it is pretty slow for the full dataset. 
Data loading is slow since it only has two cores for one Colab instance, preventing from using me too much data loader instances.
This is problematic since larger batch sizes seem to work better.
Tried to speed up the data loading, but it is a bit out of the scope of this challenge.
My next thing I would recommend is to use albumentations instead of imgaug since it has a better performance.
Nevertheless 2 processor (virtual) cores combined with a V100 is a bit unbalanced.
* Sent a mail to ask for a VM

### Wednesday 18th of May 





