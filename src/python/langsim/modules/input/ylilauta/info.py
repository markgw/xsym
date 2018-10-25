# This file is part of Pimlico
# Copyright (C) 2016 Mark Granroth-Wilding
# Licensed under the GNU GPL v3.0 - http://www.gnu.org/licenses/gpl-3.0.en.html

"""
Input reader for Ylilauta corpus.

Based on standard VRT text collection module, with a small amount of special processing added for Ylilauta.


.. seealso::

   :mod:`pimlico.modules.input.text_annotations.vrt_text`:
      Reading text from VRT files.

"""

from pimlico.core.modules.inputs import iterable_input_reader_factory
from pimlico.modules.input.text.raw_text_files.info import ModuleInfo as RawTextModuleInfo
from pimlico.modules.input.text_annotations.vrt_text.info import VRTTextOutputType
import re
from xml.sax.saxutils import unescape


id_tag = re.compile(ur">>\d+")


class YlilautaOutputType(VRTTextOutputType):
    def filter_text(self, doc):
        # Hand over to the super type for most of the processing
        sents = super(YlilautaOutputType, self).filter_text(doc)
        # XML escaping is used in the corpus: unescape
        sents = [[unescape(w) for w in sent] for sent in sents]
        # The corpus is peppered with these special ID tags that presumably refer to users
        # They're not desirable in a text corpus, so skip them
        sents = [[id_tag.sub(u"", w).strip() for w in sent] for sent in sents]
        # Remove empty words after substitution
        sents = [[w for w in sent if len(w)] for sent in sents]
        return sents


ModuleInfo = iterable_input_reader_factory(
    dict(RawTextModuleInfo.module_options, **{
        # More options may be added here
    }),
    YlilautaOutputType,
    module_type_name="ylilauta_reader",
    module_readable_name="Ylilauta VRT files",
    execute_count=True,
)
