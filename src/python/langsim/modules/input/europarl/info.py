import re

from pimlico.core.modules.inputs import iterable_input_reader_factory
from pimlico.datatypes.base import InvalidDocument
from pimlico.modules.input.text.raw_text_files.info import OutputType as RawTextOutputType, \
    ModuleInfo as RawTextModuleInfo

language_indicator_re = re.compile(r"\([A-Z]{2}\) ")


class OutputType(RawTextOutputType):
    def filter_text(self, data):
        lines = [self.filter_line(line) for line in data.splitlines()]
        lines = [line for line in lines if line is not None]
        # If a document has only a couple of lines in it, it's not a debate transcript, but just a heading
        # Better to skip these
        if len(lines) < 4:
            return InvalidDocument("reader:%s" % self.datatype_name,
                                   error_info="Too few lines in document: not a debate transcript")
        return u"\n".join(lines)

    def filter_line(self, line):
        """
        Filter applied to each line. It either returns an updated version of the line, or None, in which
        case the line is left out altogether.

        """
        if line.startswith("<"):
            # Simply filter out all lines beginning with '<', which are metadata
            return None

        # Some metadata-like text is also included at the start of lines, followed by ". - "
        if u". - " in line:
            __, __, line = line.partition(u". - ")

        # Remove -s and spaces from the start of lines
        # Not sure why they're often there, but it's just how the transcripts were formatted
        line = line.lstrip(u"- ")

        # Skip lines that are fully surrounded by brackets: they're typically descriptions of what happened
        # E.g. (Applause)
        if line.startswith(u"(") and line.endswith(u")"):
            return None

        # It's common for a speaker's first utterance to start with a marker indicating the original language
        line = language_indicator_re.sub(u"", line)
        return line


ModuleInfo = iterable_input_reader_factory(
    RawTextModuleInfo.module_options, OutputType, module_type_name="europarl_reader",
    module_readable_name="Europarl corpus reader",
)
