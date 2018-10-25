import os
import struct

from pimlico.datatypes import PimlicoDatatype, PimlicoDatatypeWriter
from pimlico.datatypes.table import get_struct


class NeuralSixgramTrainingData(PimlicoDatatype):
    """
    Prepared training data for neural sixgram model. Preparation involves extracting
    positive samples from the input corpus, shuffling them and drawing appropriate
    negative samples to match each one.

    The result is a big binary file containing the training IDs in sequence.
    Each sample consists of 12 integers: a positive 6-gram and a negative 6-gram.

    For efficiency, they're packed as C-structs.

    """
    datatype_name = "neural_sixgram_training_data"

    def __len__(self):
        return self.metadata["samples"]

    def __iter__(self):
        return self.read_rows(os.path.join(self.data_dir, "data.bin"))

    def read_rows(self, filename):
        # Compile a struct for unpacking each row
        unpacker = get_struct(2, False, 12)
        row_size = unpacker.size

        with open(filename, "rb") as f:
            while True:
                # Read data for a single row
                row_string = f.read(row_size)
                if row_string == "":
                    # Reach end of file
                    break
                try:
                    row = unpacker.unpack(row_string)
                except struct.error, e:
                    if len(row_string) < row_size:
                        # Got a partial row at end of file
                        raise IOError("found partial row at end of file: last row has byte length %d, not %d" %
                                      (len(row_string), row_size))
                    else:
                        raise IOError("error interpreting row: %s" % e)
                # Row consists of 6 ids of positive sample and 6 ids of negative
                yield tuple(row[:6]), tuple(row[6:])


class NeuralSixgramTrainingDataWriter(PimlicoDatatypeWriter):
    @property
    def data_file_path(self):
        return os.path.join(self.data_dir, "data.bin")

    def add_sample(self, sixgram, neg_sixgram):
        self.samples += 1
        self.output_file.write(self.packer.pack(*(sixgram + neg_sixgram)))

    def __enter__(self):
        obj = super(NeuralSixgramTrainingDataWriter, self).__enter__()
        # Compile a struct for packing each row
        self.packer = get_struct(2, False, 12)
        self.samples = 0
        # Open the file to output to
        self.output_file = open(self.data_file_path, "wb")
        return obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metadata["samples"] = self.samples
        super(NeuralSixgramTrainingDataWriter, self).__exit__(exc_type, exc_val, exc_tb)
        self.output_file.close()
