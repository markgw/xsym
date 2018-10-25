from pimlico.core.modules.map import skip_invalid

from pimlico.core.modules.map.multiproc import multiprocessing_executor_factory


def set_up(executor):
    # Get hold of the input char vocabs
    executor.vocab1 = executor.info.get_input("vocab1").get_data()
    executor.vocab2 = executor.info.get_input("vocab2").get_data()


def worker_set_up(worker):
    # Make things computed once at setup available on the worker for easier access
    worker.vocab1 = worker.executor.vocab1
    worker.vocab2 = worker.executor.vocab2


@skip_invalid
def process_document(worker, archive_name, doc_name, doc1, doc2):
    # Map back to text using dict
    doc1_text = [u"".join(worker.vocab1.id2token[i] for i in line) for line in doc1]
    doc2_text = [u"".join(worker.vocab2.id2token[i] for i in line) for line in doc2]
    # There should be the same number of lines in the two texts
    # Interleave them to make it easy to see what's been done
    output_text = u"\n\n".join(
        u"{}\n{}".format(
            u"".join(line1), u"".join(line2)
        ) for (line1, line2) in zip(doc1_text, doc2_text)
    )
    return output_text


ModuleExecutor = multiprocessing_executor_factory(
    process_document,
    preprocess_fn=set_up, worker_set_up_fn=worker_set_up
)
