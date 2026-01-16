# teehee-might-delete

Still in development!

todo:
- fix token segment ids
- handle sequences which exceed max token length
- implement bpe and unigram tokenizers
- support bin/h5 files?

done:
- vocab list class for storing model's vocabulary
- max match tokenizer (and most of the normalisation required)
- onnx engine (basically ort wrapper) to get embeddings for a sequence of tokens

Go to Settings > Build, Execution, Deployment > CMake and set CMake Options to '-DICU_ROOT=/opt/homebrew/opt/icu4c' (or the win/linus equivalent)


to export to onnx:
pip install "optimum[exporters]"
pip install optimum[onnxruntime]
optimum-cli export onnx --model dbmdz/bert-large-cased-finetuned-conll03-english --task token-classification bert_ner_onnx/
