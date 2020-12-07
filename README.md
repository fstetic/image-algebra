## Description
Web app made with `Flask` where users can upload their images of simple math
expressions, accepted characters are `0123456789+-x/()`, and get evaluation.

### Task
The task was to implement a simple Photomath lookalike. It was divided into three
parts.
1. Detecting characters
1. Classifying detected characters
1. Solving given math expression

### Implemented solutions
#### First task
Characters were detected by using `OpenCV` library and contour detection. 
Morphological transformations were used to get better detection. Detected
contours were cropped and reshaped to `30x30` dimensions because those are
input dimensions of the model.
#### Second task
A simple model was trained for only two epochs on a subset of Kaggle dataset of
handwritten symbols counting around 150k images. It gave good enough results with
98.54% accuracy.
#### Third task
Math expression needed to be parsed so infix to postfix parser was implemented
and postfix expression solved using Shunting-yard algorithm

### Usage
#### Web app
You can check out the app: < link > \
You can also download the whole repository and change what you like. To train
the model differently change configuration in `character_classifier.py` and then
run it.

### Conclusion
App works reasonably well for really pretty and clean inputs. I wouldn't recommend
it for real life usage, but it's a great project for learning Computer Vision.

### Dataset links
Full dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols\
Dataset subset: https://drive.google.com/file/d/1g-BnJr-QiftCtWW_s4itgrCyfE7hsieD/view?usp=sharing
