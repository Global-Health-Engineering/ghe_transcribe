# Renkulab
Renkulab is a platform that provides free computational resources for researchers. Renku is built by the [Swiss Data Science Center](https://www.datascience.ch/) with funding from the [ETH domain](https://ethrat.ch/en/). Link to their [privacy policy](https://renkulab.io/help/privacy) and [terms of use](https://renkulab.io/help/tos).

## Setup steps
- **Join Renkulab**, to run `ghe_transcribe` online
	- Sign up or Log in with
		- Edu-ID
		- Github
		- ORCID
	- Wait for email confirmation
	- Link: <a href="https://renkulab.io/api/auth/login?redirect_url=https%3A%2F%2Frenkulab.io%2F%3Frenku_login%3D1" target="_blank" rel="noopener noreferrer">renkulab join</a>

- **Join Huggingface**, to access Pyannote
	- Sign up or Log in with your email
	- Wait for email confirmation
	- Link: <a href="https://huggingface.co/join" target="_blank" rel="noopener noreferrer">huggingface join</a>

- **Accept User Conditions**, to use Pyannote
    - <a href="https://hf.co/pyannote/speaker-diarization-3.1" target="_blank" rel="noopener noreferrer">Accept conditions for speaker-diarization-3.1</a>
		- Copy and paste:
	    	- Company/university: ETH Zürich
	    	- Website: ethz.ch
    - <a href="https://huggingface.co/pyannote/segmentation-3.0" target="_blank" rel="noopener noreferrer">Accept conditions for segmentation-3.0</a>
		- Copy and paste:
	    	- Company/university: ETH Zürich
	    	- Website: ethz.ch
- **Create Access Token**, to use `ghe_transcribe`
	- Create new token
	- Token type -> `read`
	- Token name -> `ghe_transcribe`
	- Copy token to clipboard!
	- Link: <a href="https://hf.co/settings/tokens" target="_blank" rel="noopener noreferrer">https://hf.co/settings/tokens</a>

## Run ghe_transcribe
- Launch <a href="https://renkulab.io/p/nmassari/ghe-transcribe/sessions/01K7KK30KS1CS1ZW45MZCKR6TD/start?HF_HUB_DISABLE_TELEMETRY=1&HF_HUB_DISABLE_PROGRESS_BARS=1" target="_blank" rel="noopener noreferrer">Renkulab</a>
- Paste your Huggingface token and continue
- Wait for renkulab ~5-10 min
- Open ghe_transcribe folder (Left panel)
- Open app.ipynb (Left panel)
- Run all (Top panel)
- Select recommended virtual environment
- **Upload your audio file(s) and transcribe!**