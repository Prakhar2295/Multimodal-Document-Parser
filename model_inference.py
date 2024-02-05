import layoutparser as lp
from google.cloud import vision
import pandas as pd
import numpy as np
import cv2

api_key_path = r"vision-ai-api-413019-4716d3c323e9.json"

image1 = cv2.imread("")

ocr_agent = lp.GCVAgent.with_credential(api_key_path,languages = ['en'])

model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 './output/model_final.pth',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.95],
                                 label_map={0:"Table2",1: "Table1", 2: "None", 3: "None", 4:"None", 5:"None"})


image = cv2.imread("/kaggle/input/data-train/doc_img_14.jpg")
layout = model.detect(image)

text_blocks = lp.Layout([b for b in layout if b.type=="Table"])


figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])



h, w = image1.shape[:2]

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image1)

left_blocks = text_blocks.filter_by(left_interval, center=True)
left_blocks.sort(key = lambda b:b.coordinates[1])

right_blocks = [b for b in text_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# And finally combine the two list and add the index
# according to the order
text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
print(text_blocks)


for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image1))
        # add padding in each image segment can help
        # improve robustness

    layout = ocr_agent.detect(segment_image)
    block.set(text=layout, inplace=True)


filter_product_no = layout.filter_by(
    lp.Rectangle(x_1=936, y_1=969, x_2=1631, y_2=3690),
    soft_margin = {"left":10, "right":20}
)

#filter_product_no.get_texts()


filter_size_no = layout.filter_by(
    lp.Rectangle(x_1=222, y_1=969, x_2=815, y_2=3622),
    soft_margin = {"left":10, "right":20}
)
#filter_size_no.get_texts()


filter_heat_no = layout.filter_by(
    lp.Rectangle(x_1=2670, y_1=969, x_2=3021, y_2=3621),
    soft_margin = {"left":10, "right":20}
)
#filter_heat_no.get_texts()


table_dict = {
    "product_size":filter_size_no.get_texts(),
    "product_no": filter_product_no.get_texts(),
    "heat_no": filter_heat_no.get_texts()
}

df = pd.DataFrame(table_dict)
#df.head()