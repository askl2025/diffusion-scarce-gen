"""Export gate business tests using real persisted image files, no real reviews."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from experiments import coco


class ExportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        folder=self.root/'fixture';folder.mkdir()
        Image.new('RGB',(640,480),(10,20,30)).save(folder/'original.png')
        self.job={'id':'fixture','status':'done','size':[640,480],'settings':{'object_type':'dustbin'},
                  'review':{'verdict':'accepted','reviewer':'project_user','boxes':[[.25,.125,.20,.30]]}}

    def tearDown(self):self.temp.cleanup()

    def test_codex_accepted_is_excluded(self):
        self.job['review']['reviewer']='codex_visual_review'
        result=coco([self.job],self.root)
        self.assertEqual(result['images'],[]);self.assertEqual(result['annotations'],[])

    def test_unknown_missing_and_null_reviewer_are_excluded(self):
        for reviewer in ('codex_stage_validation','unknown',None,'<missing>'):
            with self.subTest(reviewer=reviewer):
                job=copy.deepcopy(self.job)
                if reviewer=='<missing>':job['review'].pop('reviewer')
                else:job['review']['reviewer']=reviewer
                result=coco([job],self.root)
                self.assertEqual(result['images'],[]);self.assertEqual(result['annotations'],[])

    def test_explicit_member_keeps_real_bbox_category_and_existing_gates(self):
        result=coco([self.job],self.root)
        self.assertEqual(len(result['images']),1);self.assertEqual(len(result['annotations']),1)
        annotation=result['annotations'][0]
        self.assertEqual(annotation['category_id'],29);self.assertEqual(annotation['bbox'],[160.0,60.0,128.0,144.0])
        self.assertEqual(annotation['area'],18432.0);self.assertEqual(annotation['reviewer'],'project_user')
        self.assertEqual(result['images'][0]['source_sha256'],hashlib.sha256((self.root/'fixture/original.png').read_bytes()).hexdigest())
        self.assertEqual({row['id'] for row in result['categories']},{1,20,23,29})
        for status,verdict,boxes in [('queued','accepted',[[.1,.1,.1,.1]]),
                                     ('done','rejected',[[.1,.1,.1,.1]]),('done','accepted',[])]:
            job=copy.deepcopy(self.job);job['status']=status;job['review'].update(verdict=verdict,boxes=boxes)
            self.assertEqual(coco([job],self.root)['annotations'],[])


if __name__=='__main__':unittest.main(verbosity=2)
