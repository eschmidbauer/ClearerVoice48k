# ClearerVoice48k

```python
from clearervoice48k.clearvoice import ClearVoice

myClearVoice_SE = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
myClearVoice_SR = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

# Perform speech enhancement
output_wav = myClearVoice_SE(input_path='samples/jfk.wav', online_write=False)
myClearVoice_SE.write(output_wav, output_path='samples/jfk-output-1.wav')
# Perform speech super-resolution
output_wav = myClearVoice_SR(input_path='samples/jfk-output-1.wav', online_write=False)
myClearVoice_SR.write(output_wav, output_path='samples/jfk-output-2.wav')
```
