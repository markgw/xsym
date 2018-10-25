import re
from string import punctuation
from pimlico.core.modules.map.multiproc import multiprocessing_executor_factory


sent_splitter = re.compile('(?<=[%s])  ' % ''.join(punctuation))
multiple_spaces = re.compile('  +')


def process_document(worker, archive_name, doc_name, doc):
    # Lowercase all text
    doc = doc.lower()
    # Strip whitespace from all lines
    lines = [l.strip() for l in doc.splitlines()]
    lines = [l for l in lines if len(l)]

    if worker.info.options["forum"]:
        # Remove the first line of every document, which is an ID
        lines = lines[1:]

    # Further split the lines where there are double spaces (which mark sentence boundaries)
    def _sentences(x):
        for line in x:
            for sent in sent_splitter.split(line):
                sent = multiple_spaces.sub(u" ", sent).strip()
                if len(sent):
                    yield sent
    lines = list(_sentences(lines))

    # Put the lines back together as a single text
    return u"\n".join(lines)


ModuleExecutor = multiprocessing_executor_factory(process_document)
