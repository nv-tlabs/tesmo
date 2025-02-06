
import os 
import json
THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "king-size bed":                           "double_bed",
    "single bed":                              "single_bed",
    "kids bed":                                "kids_bed",
    # "ceiling lamp":                            "ceiling_lamp",
    # "pendant lamp":                            "pendant_lamp",
    "bookcase/jewelry armoire":                "bookshelf",
    "tv stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
    "corner/side table":                       "table",
    "dining table":                            "table",
    "round end table":                         "table",
    "drawer chest/corner cabinet":             "cabinet",
    "sideboard/side cabinet/console table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "coffee table":                            "coffee_table",
    "loveseat sofa":                           "sofa",
    "three-seat/multi-seat sofa":              "sofa",
    "l-shaped sofa":                           "sofa",
    "lazy sofa":                               "sofa",
    "chaise longue sofa":                      "sofa",
}

THREED_FRONT_LIBRARY_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "table",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "dining table":                            "table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "table",
    "loveseat sofa":                           "sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "sofa",
    "l-shaped sofa":                           "sofa",
    "dressing table":                          "table",
    "dressing chair":                          "chair",
}

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "sofa",
    "l-shaped sofa":                           "sofa",
    "tv stand":                                "tv_stand"
}
category_dicts = dict()
for k, v in THREED_FRONT_BEDROOM_FURNITURE.items():
    if k not in category_dicts:
        category_dicts[k] = v
for k, v in THREED_FRONT_LIBRARY_FURNITURE.items():
    if k not in category_dicts:
        category_dicts[k] = v
for k, v in THREED_FRONT_LIVINGROOM_FURNITURE.items():
    if k not in category_dicts:
        category_dicts[k] = v
model_infos = json.load(open("3D_Future/model_info.json", "r"))
threeD_future_root = '/ps/project/scene_generation/ATISS_dataset/3D-FUTURE-model'
cnt = 0
for data in model_infos:
    if data['category'] is None:
        continue
    category = data['category'].lower().replace(' / ', '/')
    model_id = data['model_id']
    if category in category_dicts:
        cnt += 1
        os.makedirs(os.path.join('3D_Future_full', 'models', category_dicts[category]), exist_ok=True)
        if not os.path.exists(os.path.join('3D_Future_full', 'models', category_dicts[category], model_id)):
            os.symlink(os.path.join(threeD_future_root, model_id), os.path.join('3D_Future_full', 'models', category_dicts[category], model_id))
        if 'bed' in category_dicts[category]:
            os.makedirs(os.path.join('3D_Future_full', 'models', 'bed'), exist_ok=True)
            if not os.path.exists(os.path.join('3D_Future_full', 'models', 'bed', model_id)):
                os.symlink(os.path.join(threeD_future_root, model_id), os.path.join('3D_Future_full', 'models', 'bed', model_id))
print(cnt)