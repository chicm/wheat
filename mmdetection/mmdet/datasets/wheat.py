import os
import os.path as osp

import mmcv
import numpy as np
import pandas as pd
import random
import torch

from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose

DATA_DIR = '/relevance2-nfs/chec/data/wheat'

def get_train_folds_meta():
    SEED = 42

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    seed_everything(SEED)
    marking = pd.read_csv(f'{DATA_DIR}/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)
    #print(marking.dtypes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    return df_folds, marking

#df_folds, marking = get_train_folds_meta()


#print(df_folds.head())
#print(marking.head())

@DATASETS.register_module()
class WheatDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4),
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 ifold=0):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.ifold = ifold

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        print('>>>>>>>', len(self.data_infos), self.ann_file)
        return len(self.data_infos)

    def load_inference_annos(self):
        image_files = os.listdir(self.img_prefix)
        data_info = []
        for fn in image_files: 
            ann = {
                'filename': fn,
                'width': 1024,
                'height': 1024,
            }
            data_info.append(ann)
        return data_info

    def load_annotations(self, ann_file):
        if ann_file == 'inference': # or ann_file == 'val':
            return self.load_inference_annos()

        df_folds, marking = get_train_folds_meta()

        #return mmcv.load(ann_file)
        image_files = os.listdir(os.path.join(DATA_DIR, 'train'))
        all_img_ids = [x.split('.')[0] for x in image_files]
        no_box_images = set(all_img_ids) - set(marking.image_id.unique())

        val_ids = df_folds[df_folds['fold'] == self.ifold].index.values.tolist() + list(no_box_images)
        train_ids = df_folds[df_folds['fold'] != self.ifold].index.values.tolist()

        if ann_file == 'train':
            img_ids = train_ids
        elif ann_file == 'val':
            img_ids = val_ids
        else:
            raise ValueError('ann file name')
        
        data_info = []
        for img_id in img_ids:
            records = marking[marking['image_id'] == img_id]
            if len(records) < 1:
                labels = np.array([2.])
                boxes = np.array([[0., 0., 1., 1.]])
                source = 'unknown'
            else:
                boxes = records[['x', 'y', 'w', 'h']].values
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

                labels = np.array([1] * len(boxes))
                source = records['source'].values[0]

            ann = {
                'filename': f'{img_id}.jpg',
                'width': 1024,
                'height': 1024,
                'ann': {
                    'bboxes': boxes.astype(np.float32),
                    'labels': labels.astype(np.long),
                    'source': source
                    #'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    #'labels_ignore': <np.ndarray> (k, 4)  (optional field)
                }
            }
            data_info.append(ann)
        return data_info

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        if self.ann_file == 'inference':
            return None
        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_subset_by_classes(self):
        return self.data_infos

    def format_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        print('VALIDATING...')
        print(len(results), len(results[0]), results[0][1].shape)
        print(results[0][1][:5])
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]

        import pickle
        print('saving')
        with open('val_ann_2.pkl', 'wb') as f:
            pickle.dump(self.data_infos, f)
        with open('val_preds_2.pkl', 'wb') as f:
            pickle.dump(results, f)
        print('saved')


        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            ap_results = []
            for iou_thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                ap_results.append(mean_ap)
            eval_results['mAP'] = np.mean(ap_results)
            print('>>>>>MAP:', np.mean(ap_results))
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
