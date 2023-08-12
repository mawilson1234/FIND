# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adopted from fairseq https://github.com/pytorch/fairseq/

import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
import json
import os

from time import sleep
# from glob import glob

# there are some timing issues when loading the initial checkpoint
# we'll allow up to 10 retries before exiting
MAX_RELOAD_TRIES = 10
WAIT_BETWEEN_RELOAD_TRIES = 120/MAX_RELOAD_TRIES # wait up to 2 minutes total

def main(args):
    assert args.path is not None, '--path required for generation!'
    args.beam = args.nbest = 1
    args.max_tokens = int(1e4)

    utils.import_user_module(args)

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    src_dict = getattr(task, 'source_dictionary', None)
    tgt_dict = task.target_dictionary
    
    n_tries = 0
    while n_tries < MAX_RELOAD_TRIES:
        try:
            models, _model_args = checkpoint_utils.load_model_ensemble(
                args.path.split(':'),
                arg_overrides=eval(args.model_overrides),
                task=task,
            )
            break
        except Exception as e:
            # many kinds of exceptions can happen, EOFError, FileNotFoundError, OSError, RuntimeError, etc.
            # so we're trying to be general here
            if n_tries == MAX_RELOAD_TRIES - 1:
                raise e
            
            print(f'Unable to load {str(args.path.split(":"))!r} on {n_tries} try. Retrying...')
            sleep(WAIT_BETWEEN_RELOAD_TRIES)
            n_tries += 1

    # models, _model_args = checkpoint_utils.load_model_ensemble(
    #     args.path.split(':'),
    #     arg_overrides=eval(args.model_overrides),
    #     task=task,
    # )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=args.beam,
            need_attn=False
        )
        model.cuda()

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    generator = task.build_generator(args)
    
    output_dir = os.path.dirname(args.path)
    with progress_bar.build_progress_bar(args, itr) as t, \
         open(f'{output_dir}/generated-{args.gen_subset}.json', 'wt', encoding='utf8') as out_file:
        for sample in t:
            sample = utils.move_to_cuda(sample)
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            
            # handle some weird AssertionErrors that particular random seeds cause for particular architectures+hyperparams.
            # for now, just skip them and it'll be taken into account in the %s. 
            # appears to be due to nans, related to the older version of fairseq this repo uses.
            # see https://github.com/facebookresearch/fairseq/issues/2087
            try:
                hypos = task.inference_step(generator, models, sample, prefix_tokens)
            except AssertionError as e:
                print(e)
                print('AssertionError was raised. Skipping this sample for this seed.')
                continue
                
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

            for i, sample_id in enumerate(sample['id'].tolist()):
                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())

                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""

                # Process top predictions
                hypo = hypos[i][0]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                result = dict(src=src_str, pred=hypo_str, src_len=len(src_str.split()), pred_len=len(hypo_str.split()))
                result_line = json.dumps(result)
                json.dump(result, out_file, ensure_ascii=False)
                out_file.write('\n')
                
                print(result_line)
    
    # remove unneeded checkpoints
    # checkpoints = glob(f'{output_dir}/*.pt')
    # for file in checkpoints:
    #    try:
    #        os.remove(file)
    #    except Exception:
    #        pass


def cli_main(args):
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=args)
    main(args)


if __name__ == '__main__':
    import sys
    cli_main(sys.argv[1:])
