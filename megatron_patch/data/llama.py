# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from megatron_patch.tokenizer import get_tokenizer

"""
PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task,'
             ' paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}'
             '\n\n### Input:\n{input}\n\n### Response:'),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }

PROMPT_DICT = {
    'prompt_input': ('[INST]{instruction} {input}\n[/INST]\n'),
    'prompt_no_input':('[INST]{instruction}\n[/INST]\n'),
}

PROMPT_DICT = {
    'prompt_input': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n',
    'prompt_no_input':'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
}
"""

PROMPT_DICT = {
    'prompt_input': ('{instruction} {input}'),
    'prompt_no_input': ('{instruction}'),
}


class LLamaRawDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, max_padding_length, split='train'):
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        if "-Pretrain" in args.dataset:
            self.max_padding_length = max_padding_length + 1
        else:
            self.max_padding_length = max_padding_length

        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=16,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            desc="Running Encoding"
        )

        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []

        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            if self.tokenizer.eos_token_id not in inputs: continue
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """

        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

        sources = []
        if 'input' not in examples:
            if 'instruction' in examples:
                for instruction in examples['instruction']:
                    sources.append(prompt_no_input.format_map({"instruction": instruction}))
            elif 'query' in examples:
                for query in examples['query']:
                    sources.append(prompt_no_input.format_map({"instruction": query}))
        else:
            if 'instruction' in examples:
                for instruction, minput in zip(examples['instruction'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": instruction, "input": minput}))
            elif 'query' in examples:
                for query, minput in zip(examples['query'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": query, "input": minput}))

        if 'output' in examples:
            key = 'output'
        elif 'content' in examples:
            key = 'content'
        elif 'response' in examples:
            key = 'response'

        targets = [
            example + self.tokenizer.eos_token
            for example in examples[key]
        ]

        examples_raw = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self.tokenize(strings, self.tokenizer)
            for strings in (examples_raw, sources)
        ]
        input_ids = examples_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized['input_ids_lens']):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        Tokenize a list of strings.
        Args:
            strings (List[str]): a list of strings to be tokenized
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the tokenized strings
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample


class LLamaIdxMapDataset(torch.utils.data.Dataset):
    """LLAMA dataset class for mmap format data"""

    def __init__(self,
                 name,
                 data_prefix,
                 documents,
                 indexed_dataset,
                 num_samples,
                 seed,
                 max_padding_length,
                 return_doc_ids=False):

        args = get_args()
        self.tokenizer = get_tokenizer()
        self.max_padding_length = max_padding_length

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids
        self.split = args.split
        # Checks
        assert np.min(documents) >= 0
        try:
            assert np.max(documents) < indexed_dataset.sizes().shape[0]
        except:

            assert np.max(documents) < len(indexed_dataset)

        # Build index mappings.
        try:
            from megatron.data.gpt_dataset import _build_index_mappings
        except:
            _build_index_mappings = self._build_index_mappings

        try:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
                _build_index_mappings(self.name, data_prefix,
                                      documents, self.indexed_dataset.sizes,
                                      num_samples, self.max_padding_length, seed)
        except:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = \
                _build_index_mappings(self.name, data_prefix,
                                      documents, self.indexed_dataset.sizes,
                                      self.split, num_samples, self.max_padding_length, seed,
                                      data_cache_path=None)
            

    def _build_shuffle_idx(num_samples, total_size, np_rng):
        """Build the range [0, size) and shuffle."""
        print(' > building shuffle index with split [0, {}) and [{}, {}) '
            '...'.format(num_samples, num_samples, total_size), flush=True)

        dtype_ = np.uint32
        if total_size >= (np.iinfo(np.uint32).max - 1):
            dtype_ = np.int64

        shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                    step=1, dtype=dtype_)
        np_rng.shuffle(shuffle_idx_first)
        if num_samples == total_size:
            return shuffle_idx_first

        shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                    step=1, dtype=dtype_)
        np_rng.shuffle(shuffle_idx_last)

        return np.concatenate((shuffle_idx_first, shuffle_idx_last))


    def _build_doc_idx(self, documents, num_epochs, np_rng, separate_last_epoch):
        """Build an array with length = number-of-epochs * number-of-dcuments.
        Each index is mapped to a corresponding document."""
        if not separate_last_epoch or num_epochs == 1:
            doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
            doc_idx[:] = documents
            doc_idx = doc_idx.reshape(-1)
            doc_idx = doc_idx.astype(np.int32)
            np_rng.shuffle(doc_idx)
            return doc_idx

        doc_idx_first = self._build_doc_idx(documents, num_epochs-1, np_rng, False)
        doc_idx_last = self._build_doc_idx(documents, 1, np_rng, False)
        return np.concatenate((doc_idx_first, doc_idx_last))


    def _num_tokens(self, documents, sizes):
        """Total number of tokens in the dataset."""
        return np.sum(sizes[documents])


    def _num_epochs(self, tokens_per_epoch, seq_length, num_samples):
        """Based on number of samples and sequence lenght, calculate how many
        epochs will be needed."""
        num_epochs = 0
        total_tokens = 0
        while True:
            num_epochs += 1
            total_tokens += tokens_per_epoch
            # -1 is because we need to retrieve seq_length + 1 token each time
            # but the last token will overlap with the first token of the next
            # sample except for the last sample.
            if ((total_tokens - 1) // seq_length) >= num_samples:
                return num_epochs


    def _build_index_mappings(self, name, data_prefix, documents, sizes,
                          splits_string, num_samples, seq_length, seed,
                          *,
                          data_cache_path):
        """Build doc-idx, sample-idx, and shuffle-idx.
        doc-idx: is an array (ordered) of documents to be used in training.
        sample-idx: is the start document index and document offset for each
        training sample.
        shuffle-idx: maps the sample index into a random index into sample-idx.
        """

        import os, hashlib, time
        from megatron.training  import print_rank_0
        from megatron.core import mpu
        # Number of tokens in each epoch and number of required epochs.
        tokens_per_epoch = self._num_tokens(documents, sizes)
        num_epochs = self._num_epochs(tokens_per_epoch, seq_length, num_samples)

        # rng state
        np_rng = np.random.RandomState(seed=seed)

        # Filename of the index mappings.
        desc = "GPT Dataset\n\n"
        desc += f"Data prefix {data_prefix}\n"
        desc += f"Dataset name {name}\n"
        desc += f"Number of samples {num_samples}\n"
        desc += f"Sequence length {seq_length}\n"
        desc += f"Random seed {seed}\n"
        desc += f"Split {splits_string}\n"
        desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
        desc_filename = desc_hash + ".dsc"
        doc_idx_filename = desc_hash + '_doc_idx.npy'
        sample_idx_filename = desc_hash + '_sample_idx.npy'
        shuffle_idx_filename = desc_hash + '_shuffle_idx.npy'

        # Look for cache in main data dir first to avoid unnecessary
        # duplication, then look in data-cache-path if specified,
        # If nothing is found, use the last path looked in
        build_indices = True
        prefixes = [os.path.join(os.path.dirname(data_prefix), 'index-cache')]
        if data_cache_path is not None:
            prefixes.append(data_cache_path)
        for prefix in prefixes:
            idx_path = {
                'desc': os.path.join(prefix, desc_filename),
                'doc': os.path.join(prefix, doc_idx_filename),
                'sample': os.path.join(prefix, sample_idx_filename),
                'shuffle': os.path.join(prefix, shuffle_idx_filename)
            }
            for f in idx_path.values():
                if not os.path.isfile(f):
                    break
            else:
                # Found our files!
                build_indices = False
                break
        data_cache_dir = os.path.dirname(idx_path['desc'])
        data_cache_success = True

        # Build the indexed mapping if not exist.
        if build_indices and torch.distributed.get_rank() == 0:
            print_rank_0(' > WARNING: could not find index map files, building '
                        'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                    'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                        num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples <= (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                    int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                            'than 80% of number of samples per epoch ({}), '\
                            'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                            'than 80% of number of samples per epoch ({}), '\
                            'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)


            try:
                os.makedirs(data_cache_dir, exist_ok=True)

                # description
                with open(idx_path['desc'], 'wt') as fd:
                    fd.write(desc)

                # doc-idx.
                start_time = time.time()
                doc_idx = self._build_doc_idx(documents, num_epochs, np_rng,
                                        separate_last_epoch)
                np.save(idx_path['doc'], doc_idx, allow_pickle=True)
                print_rank_0(' > elasped time to build and save doc-idx mapping '
                            '(seconds): {:4f}'.format(time.time() - start_time))
                # sample-idx.
                start_time = time.time()
                # Use C++ implementation for speed.
                # First compile and then import.
                from megatron.data import helpers
                assert doc_idx.dtype == np.int32
                assert sizes.dtype == np.int32
                sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                    num_epochs, tokens_per_epoch)
                np.save(idx_path['sample'], sample_idx, allow_pickle=True)
                print_rank_0(' > elasped time to build and save sample-idx mapping '
                            '(seconds): {:4f}'.format(time.time() - start_time))
                # shuffle-idx.
                start_time = time.time()
                # -1 is due to data structure used to retieve the index:
                #    sample i --> [sample_idx[i], sample_idx[i+1])
                if separate_last_epoch:
                    num_samples_ = num_samples_from_epochs_minus_one
                else:
                    num_samples_ = sample_idx.shape[0] - 1
                shuffle_idx = self._build_shuffle_idx(num_samples_,
                                                sample_idx.shape[0] - 1, np_rng)
                np.save(idx_path['shuffle'], shuffle_idx, allow_pickle=True)
                print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                            ' (seconds): {:4f}'.format(time.time() - start_time))
            except OSError:
                print(f'There was an error trying to create the data cache directory ({data_cache_dir})')
                print('or a file in it. This defaults to a directory "index-cache" within the directory')
                print('the data files are in and can be set with the --data-cache-path argument. Please')
                print('ensure you have write access to this directory or specify one that you do have')
                print('write access to.')
                data_cache_success = False

        counts = torch.cuda.LongTensor([data_cache_success])
        torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
        if counts[0].item() != (
            torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())):
            print_rank_0("Data index creation unsuccessful, exiting.")
            exit()

        # Load mappings.
        start_time = time.time()
        print_rank_0(f" > loading doc-idx mapping from {idx_path['doc']}")
        doc_idx = np.load(idx_path['doc'], allow_pickle=True, mmap_mode='r')

        print_rank_0(f" > loading sample-idx mapping from {idx_path['sample']}")
        sample_idx = np.load(idx_path['sample'], allow_pickle=True, mmap_mode='r')

        print_rank_0(f" > loading shuffle-idx mapping from {idx_path['shuffle']}")
        shuffle_idx = np.load(idx_path['shuffle'], allow_pickle=True, mmap_mode='r')

        print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
            time.time() - start_time))
        print_rank_0('    total number of samples: {}'.format(
            sample_idx.shape[0]))
        print_rank_0('    total number of epochs: {}'.format(num_epochs))

        return doc_idx, sample_idx, shuffle_idx, desc, desc_hash



    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []

        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [
                self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                         offset=offset_f)
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l],
                                         length=offset_l + 1))
            sample = np.concatenate(sample_list)

        tokens = sample.tolist()
        sample = []
        sample.append(np.array(tokens))
        sample.append(np.array(tokens))

        return self.gpt_convert_example_to_feature(sample)

    def gpt_convert_example_to_feature(self, sample):
        input_ids, labels = sample
        loss_mask = np.ones(labels.shape, dtype=np.int64)
        loss_mask[labels == self.tokenizer.bos_token_id] = 0
        loss_mask[labels == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask
        }

        return train_sample