import argparse
import json
import warnings
warnings.simplefilter("ignore", ResourceWarning)



CODETR_THS_PER_CLASS = {1: 0.4, 0: 0.5, 2: 0.5, 3: 0.4, 4: 0.45}
GT_PATHS = {
    "val": "/aicity/data/FishEye8K/val/val.json",
    "test": "/aicity/data/FishEye8K/test/test.json",
    "test-challenge": "/aicity/data/FishEye8K/test_challenge.json",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="json path")
    parser.add_argument("--split", type=str, choices=["val", "test", "test-challenge"], default="val", help="split name")
    parser.add_argument("--skip_metrics", default=False, action="store_true", help="if True, metrics calculation will be skipped")
    args = parser.parse_args()
    
    
    results = json.load(open(args.detections))
    results = [x for x in results if x['score'] >= CODETR_THS_PER_CLASS[x['category_id']]]

    filtered_path = args.detections.replace('.json', '_mapped.json')
    print(f"Saving filtered detections: {filtered_path}")
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
      
    if not args.skip_metrics and args.split!="test-challenge":
        print(f"Running evaluation: {filtered_path}")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval_modified import COCOeval
        
        coco_gt = COCO(GT_PATHS[args.split])
        coco_dt = coco_gt.loadRes(filtered_path)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        