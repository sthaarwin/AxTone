# AxTone API Reference

## Audio Processing Module

### `process_audio(file_path, **kwargs)`
Processes an audio file and extracts key features.

**Parameters:**
- `file_path` (str): Path to the audio file
- `**kwargs`: Additional processing parameters

**Returns:**
- `dict`: Extracted audio features

---

## Tab Generation Module

### `generate_tab(audio_features, instrument='guitar')`
Generates musical tablature from audio features.

**Parameters:**
- `audio_features` (dict): Audio features extracted by the processing module
- `instrument` (str): Target instrument (default: 'guitar')

**Returns:**
- `TabNotation`: Object containing the generated tablature

---

## Utility Functions

### `export_tab(tab, format='txt', output_path=None)`
Exports tablature to various formats.

**Parameters:**
- `tab` (TabNotation): The tablature to export
- `format` (str): Output format ('txt', 'pdf', 'midi')
- `output_path` (str, optional): Path for the output file

**Returns:**
- `bool`: Success status