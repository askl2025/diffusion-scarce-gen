"""Owned CPU preprocessing child. JSON lines on stdin/stdout, no network listener."""
import argparse,json,os,sys
from pathlib import Path
from PIL import Image
from scene_analysis import write_scene_files

os.environ['CUDA_VISIBLE_DEVICES']=''
parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);args=parser.parse_args()
for line in sys.stdin:
    try:
        request=json.loads(line);folder=Path(request['folder'])
        with Image.open(folder/'input.png') as image:meta=write_scene_files(image.convert('RGB'),folder,args.cache)
        response={'id':request['id'],'scene':meta}
    except Exception as error:
        response={'id':request.get('id') if 'request' in locals() else None,'error':str(error)}
    print(json.dumps(response,ensure_ascii=False),flush=True)
