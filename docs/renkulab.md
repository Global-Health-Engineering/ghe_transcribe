# Renkulab
## Prerequisites
- **Join Renkulab**, a [Swiss Data Science Center](https://www.datascience.ch/) initiative partnered with EPFL and ETH Zürich
	- [renkulab join](https://renkulab.io/api/auth/login?redirect_url=https%3A%2F%2Frenkulab.io%2F%3Frenku_login%3D1)
	- Sign up or Log in with
		- Edu-ID
		- Github
		- ORCID
	- Email confirmation

- **Join Huggingface**, to access Pyannote
	- [huggingface join](https://huggingface.co/join)
	- Email confirmation
- **Accept User Conditions**, to use Pyannote
    - [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
	    - Company/university: ETH Zürich
	    - Website: [ethz.ch](ethz.ch)
    - [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
	    - Company/university: ETH Zürich
	    - Website: [ethz.ch](ethz.ch)
- **Create Access Token**, to use `ghe_transcribe`
	- [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
	- Create new token
	- Token type -> `read`
	- Token name -> `ghe_transcribe`
	- Copy token to clipboard!

## Run ghe_transcribe
- Launch [Renkulab](https://renkulab.io/p/nmassari/ghe-transcribe/sessions/01K7KK30KS1CS1ZW45MZCKR6TD/start?HF_HUB_DISABLE_TELEMETRY=1&HF_HUB_DISABLE_PROGRESS_BARS=1)
- Paste your Huggingface token and continue
- Wait for renkulab ~5-10 min
- Open app.ipynb (Left panel)
- Run all (Top panel)
- Upload your audio file(s) and transcribe