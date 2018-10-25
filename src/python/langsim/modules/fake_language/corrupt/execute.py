import random
import copy
import json

import numpy as np
from pimlico.core.modules.map import skip_invalid
from pimlico.core.modules.execute import ModuleExecutionError
from pimlico.core.modules.map.multiproc import multiprocessing_executor_factory
from pimlico.datatypes.dictionary import DictionaryWriter
from pimlico.datatypes.files import NamedFileWriter


def set_up(executor):
    # Get hold of the input char vocab
    vocab = executor.info.get_input("vocab").get_data()
    executor.input_vocab = vocab
    # We expect one of the characters to be a space, which we use for various purposes
    try:
        executor.space_id = vocab.token2id[u" "]
    except KeyError:
        # NB This could be made optional, which would involve changing the behaviour of the
        # corruption a bit, but for now we require it
        raise ModuleExecutionError("input vocab didn't include a space. At the moment, we need to identify "
                                   "the ID of space for some of the corruptions")
    # If OOV is in the vocabulary, exclude it from mappings
    oov_ids = []
    if u"OOV" in vocab.token2id:
        oov_ids.append(vocab.token2id[u"OOV"])
    # Also look up the frequency of each char, so we can select from the unigram distribution
    freqs = executor.info.get_input("frequencies").array.astype(np.float32)
    # Set the probability of space to zero, so we don't map to spaces and split up words
    freqs[executor.space_id] = 0.
    probs = freqs / freqs.sum()
    executor.unigram_dist = probs

    new_vocab = copy.deepcopy(vocab)

    # Choose random mappings to apply
    char_map_prop = executor.info.options["char_map_prop"]
    if char_map_prop > 0.:
        # Number of characters to apply a mapping to
        # Choose the characters to map from
        # Don't map spaces
        chars_to_map, actual_char_map_prop = sample_to_frequency(freqs, char_map_prop,
                                                                 exclude=[executor.space_id]+oov_ids)
        executor.log.info("Chose {} characters to apply a mapping to. Target prop={}, actual expected prop={}"
                          .format(len(chars_to_map), char_map_prop, actual_char_map_prop))
        # Choose characters to map to
        # Don't map to one of the other characters we're mapping from, as they'll be removed from the new vocab
        # Don't map to spaces either, as lots of words will get split up
        targ_chars = list(set(range(len(new_vocab))) - set(chars_to_map) - {executor.space_id} - set(oov_ids))
        random.shuffle(targ_chars)
        if len(targ_chars) < len(chars_to_map):
            # Repeat target chars if there aren't enough
            targ_chars *= (len(chars_to_map) / len(targ_chars)) + 1
        # Put together the mapping
        executor.char_map = dict(zip(chars_to_map, targ_chars))
        # For debugging and output, prepare dict of mapped chars
        mapped_chars = dict(
            (vocab.id2token[src], vocab.id2token[trg])
            for (src, trg) in executor.char_map.items()
        )
        executor.log.info(u"Mapping and removing characters: {}".format(
            u", ".join(u"{}>{}".format(src, trg) for (src, trg) in mapped_chars.items())
        ))
        # Remove mapped char IDs from the dictionary
        new_vocab.filter_tokens(bad_ids=chars_to_map)
    else:
        # No char mapping
        executor.char_map = None
        chars_to_map = []
        mapped_chars = {}
        actual_char_map_prop = 0.

    char_split_prop = executor.info.options["char_split_prop"]
    if char_split_prop > 0.:
        # Choose characters to split
        try:
            chars_to_split, actual_char_split_prop = sample_to_frequency(
                freqs, char_split_prop,
                exclude=[executor.space_id] + chars_to_map + oov_ids
            )
        except ProportionCannotBeReached:
            raise ModuleExecutionError(
                "Characters left in vocab after mapping do not cover a large enough proportion of tokens to "
                "reach the requested proportion for splitting ({})".format(char_split_prop)
            )
        executor.log.info("Chose {} characters to apply splitting to. Target prop={}, actual expected prop={}"
                          .format(len(chars_to_split), char_split_prop, actual_char_split_prop))
        # Build the char split dictionary in terms of characters for now, so we can store it in terms
        # of IDs in the new vocab once they're finalized
        char_splits_chars = {}
        for src_char_id in chars_to_split:
            new_char = get_new_character(new_vocab.token2id.keys())
            # Add the new char to the new vocab that will be output
            new_vocab.add_term(new_char)
            char_splits_chars[vocab.id2token[src_char_id]] = new_char

        executor.log.info(u"Splitting: mapping half of occurrences to new chars: {}".format(
            u", ".join(u"{}>{}".format(src, trg) for (src, trg) in char_splits_chars.items())
        ))
    else:
        char_splits_chars = {}
        actual_char_split_prop = 0.

    # All IDs will need to be mapped to the new vocabulary, if it's changed at all
    # Always compactify to be on the safe side
    new_vocab.compactify()
    executor.id_map = dict(
        (old_id, new_vocab.token2id[vocab.id2token[old_id]]) for old_id in vocab.id2token.keys()
        # Skip chars that have been removed: they will all be mapped to something else
        if vocab.id2token[old_id] in new_vocab.token2id
    )

    # Rebuild the char split map so that all IDs are from the new vocab
    # The target IDs must be from the new, so we apply the mapping after mapping to new vocab IDs
    if len(char_splits_chars):
        char_splits = dict(
            (new_vocab.token2id[src_tok], new_vocab.token2id[trg_tok])
            for (src_tok, trg_tok) in char_splits_chars.items()
        )
    else:
        char_splits = None
    executor.char_splits = char_splits

    # Some corruptions might change the vocabulary, removing or adding characters
    # That should happen here, then we write the new vocabulary
    # For now, we just write out the input vocab
    executor.log.info("Outputting new vocabulary")
    with DictionaryWriter(executor.info.get_absolute_output_dir("vocab")) as writer:
        writer.data = new_vocab

    # Output the parameters that we're using, so that later modules can check how the data was generated
    with NamedFileWriter(executor.info.get_absolute_output_dir("corruption_params"),
                         executor.info.get_output("corruption_params").filename) as writer:
        writer.write_data(json.dumps({
            "char_map_prop": executor.info.options["char_map_prop"],
            "char_split_prop": executor.info.options["char_split_prop"],
            "char_subst_prop": executor.info.options["char_subst_prop"],
            "actual_char_map_prop": actual_char_map_prop,
            "actual_char_split_prop": actual_char_split_prop,
        }))

    # Output a JSON file containing the mappings that we've applied, so we can easily check later
    # how well the learning method recovered the applied mappings
    with NamedFileWriter(executor.info.get_absolute_output_dir("mappings"),
                         executor.info.get_output("mappings").filename) as writer:
        writer.write_data(json.dumps({
            "mapped": mapped_chars,
            "split": char_splits_chars,
            # Also output the parameters used, so we can easily do things like plot results of grid searches
            "char_subst_prop": executor.info.options["char_subst_prop"],
            "char_split_prop": char_split_prop,
            "char_map_prop": char_map_prop,
            "actual_char_map_prop": actual_char_map_prop,
            "actual_char_split_prop": actual_char_split_prop,
        }))

    # Prepare a list of pairs that we expect to have a close correspondence to each other
    # The first item is in the uncorrupted data, the second in the corrupted data
    # This includes:
    #   1. direct mappings
    #   2. split characters with the new character added for them
    #   3. split characters with themselves (they maintain their identity half the time)
    #   4. other, unmapped characters with themselves
    close_pairs = \
        list(mapped_chars.items()) + \
        list(char_splits_chars.items()) + \
        [(c, c) for c in char_splits_chars.keys()] + \
        [(c, c) for c in vocab.token2id.keys() if c not in mapped_chars and c not in char_splits_chars]

    with NamedFileWriter(executor.info.get_absolute_output_dir("close_pairs"),
                         executor.info.get_output("close_pairs").filename) as writer:
        writer.write_data(json.dumps(close_pairs))


def worker_set_up(worker):
    # Make things computed once at setup available on the worker for easier access
    worker.unigram_dist = worker.executor.unigram_dist
    worker.input_vocab = worker.executor.input_vocab
    worker.space_id = worker.executor.space_id
    worker.char_map = worker.executor.char_map
    worker.id_map = worker.executor.id_map
    worker.char_splits = worker.executor.char_splits
    # Pull out some options for ease of access
    worker.char_subst_prop = worker.info.options["char_subst_prop"]


def corrupt_line(line, worker):
    new_line = list(line)

    if worker.char_subst_prop > 0.:
        # Randomly select characters to replace
        for char_pos, char in enumerate(new_line):
            if char != worker.space_id and random.random() <= worker.char_subst_prop:
                # This one should be substituted
                # Pick a new character ID from the unigram dist
                new_char = np.random.choice(worker.unigram_dist.shape[0], p=worker.unigram_dist)
                new_line[char_pos] = new_char

    if worker.char_map is not None:
        # Map the character if it's in the map, otherwise leave it
        new_line = [worker.char_map.get(i, i) for i in new_line]

    # No matter what else we do, we map to the new vocab, which usually has a new set of IDs
    new_line = [worker.id_map.get(i, i) for i in new_line]

    # Apply char splitting after mapping to the new vocab's IDs
    if worker.char_splits is not None:
        new_line = [split_char(i, worker.char_splits) for i in new_line]

    return new_line


def split_char(char, char_splits):
    if char in char_splits:
        # This char is subject to splitting: choose whether to split, with 50% prob
        if random.random() <= 0.5:
            return char_splits[char]
    return char


CHAR_LIKE_RANGES = [
    (48, 90),    # Capital letters
    (97, 122),   # Lowercase letters
    (192, 214),  # Other letters + diacritics, uppercase
    (216, 222),  # Other letters + diacritics, uppercase
    (223, 246),  # Other letters + diacritics, lowercase
    (248, 255),  # Other letters + diacritics, lowercase
]
CHAR_IDS = sum([
    list(range(a, b+1)) for (a, b) in CHAR_LIKE_RANGES
], [])


def get_new_character(used):
    """
    Choose a new unicode character from unicode ranges that correspond to the sort of latin
    characters used in languages (e.g. various diacritics). Ensure that the chosen character
    is not in the given set.

    """
    while True:
        new_char = unichr(random.choice(CHAR_IDS))
        if new_char not in used:
            return new_char


class ProportionCannotBeReached(Exception):
    pass


def sample_to_frequency(frequencies, target_prop, exclude=[]):
    """
    Choose some indices of types (e.g. characters from vocabulary), sampled at random such
    that the expected total proportion of tokens they cover is >= ``target_prop``.

    Returns a list of indices and the actual expected proportion of tokens that the
    sample results in.

    """
    props = frequencies / frequencies.sum()
    # Never choose something with a frequency of 0
    indices = [i for i in range(len(props)) if frequencies[i] > 0 and i not in exclude]
    random.shuffle(indices)
    cum_freqs = np.cumsum([props[i] for i in indices])
    if not np.any(cum_freqs >= target_prop):
        # With the excluded indices taken into account, it's impossible to reach the required proportion of tokens
        raise ProportionCannotBeReached()
    # Take indices until the cumulative proportion goes over the target prop
    # Argmax over booleans gives you the first true value
    last_index = np.argmax(cum_freqs >= target_prop)
    return indices[:last_index+1], float(cum_freqs[last_index+1])


@skip_invalid
def process_document(worker, archive_name, doc_name, doc):
    corrupted_lines = [
        corrupt_line(line, worker) for line in doc
    ]
    return corrupted_lines


ModuleExecutor = multiprocessing_executor_factory(
    process_document,
    preprocess_fn=set_up, worker_set_up_fn=worker_set_up
)
