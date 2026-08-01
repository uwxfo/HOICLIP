import unittest

import torch

from engine import _gather_hoi_eval_batch_on_main, _prepare_hoi_eval_batch


class EvalMemoryTest(unittest.TestCase):
    def test_prepare_eval_batch_drops_unused_image_tensors(self):
        results = [{
            'hoi_scores': torch.ones(2, 3),
            'clip_visual': torch.ones(2, 4),
            'clip_logits': torch.ones(1, 3),
        }]
        targets = [{
            'id': torch.tensor(7),
            'boxes': torch.ones(1, 4),
            'clip_inputs': torch.ones(3, 224, 224),
            'raw_img': torch.ones(3, 8, 8),
        }]

        compact_results, compact_targets = _prepare_hoi_eval_batch(results, targets)

        self.assertEqual(set(compact_results[0]), {'hoi_scores'})
        self.assertEqual(set(compact_targets[0]), {'id', 'boxes'})

    def test_prepare_eval_batch_keeps_visual_feature_only_when_requested(self):
        results = [{'clip_visual': torch.ones(2, 4), 'clip_logits': torch.ones(1, 3)}]

        compact_results, _ = _prepare_hoi_eval_batch(
            results, [], keep_clip_visual=True,
        )

        self.assertEqual(set(compact_results[0]), {'clip_visual'})

    def test_single_process_gather_preserves_results_and_targets(self):
        preds = []
        gts = []
        results = [{'hoi_scores': torch.ones(2, 3)}]
        targets = [{'id': torch.tensor(7)}]

        _gather_hoi_eval_batch_on_main(results, targets, preds, gts)

        self.assertIs(preds[0], results[0])
        self.assertIs(gts[0], targets[0])


if __name__ == '__main__':
    unittest.main()
