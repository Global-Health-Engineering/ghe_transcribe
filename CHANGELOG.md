## [0.7.0](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.6.1...v0.7.0) (2025-07-10)


### Features

* accept all audio formats ([264781f](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/264781f3f400416fe49d9a03a56ff7a2b37b4b28))


### Bug Fixes

* automatic creation of output dir [#44](https://github.com/Global-Health-Engineering/ghe_transcribe/issues/44), make sure output/ exists ([ba8c048](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/ba8c04895484e815be44ac429037937a7dbfcc71))

## [0.6.1](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.6.0...v0.6.1) (2025-06-27)


### Bug Fixes

* build error ([3813d2b](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/3813d2bb7a5e93d658ebc7b0824dbb0d760ec929))
* optimize CI workflow to reduce GitHub Actions minutes on dev branch ([cadfaa6](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/cadfaa6fe92263511d9181809fd684af223783a6))
* uv lock update, ci thinning in dev ([5b32010](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/5b32010d0bec4a2edf158ac8b6d4d15f817bf787))

## [0.6.0](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.5.2...v0.6.0) (2025-06-25)


### Features

* add comprehensive audio conversion tests for MP3 and M4A to WAV ([52bccf1](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/52bccf16cb631041693fd74f244c8ef637ab08c7))
* allow output area to expand fully without height restrictions ([37ff11a](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/37ff11a91b03b564a5d7285f759e27366c59c886))
* center all widgets with auto margins in common layout ([9991fd0](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/9991fd04060fd8c2d3adc7e09012c02fd279b1b4))
* enable automatic width sizing with horizontal scrolling ([a81a47d](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/a81a47d6d70e24121270a441fca42c869a83e063))
* improve Jupyter widget layout and output display ([d8e3909](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/d8e3909f332e2867f5151164fcb95ee5807f1b7f))
* remove width restrictions from output container ([228515a](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/228515a395b943d07d098344d9536b432062df7c))
* replace CSV output with TXT format and change speaker labels to SXX ([4c48f1e](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/4c48f1e0643744f094bced5127c0aabbb03c85ad))
* separate output area into expandable horizontal box ([7acb764](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/7acb764da45f7e426ad0163b5523fd11b4748b7c))


### Bug Fixes

* remove publish to pypi ([d8edcd0](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/d8edcd0b99b11d05964ba90fc556127d62c07bd8))
* restore horizontal and vertical scrolling to output area ([3331dd2](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/3331dd2c3704abe89964e2f6ab2877f5365a91fb))
* restore output widget with improved styling for better UX ([9c59fa4](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/9c59fa4da50f6faa3378325d2696298b3d751ab0))


### Reverts

* simplify output area to 90% width without scrolling ([487d5b3](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/487d5b3063a8c0cf7b5d2d8f9b08e22b5994c3f3))

## [0.5.2](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.5.1...v0.5.2) (2025-06-23)


### Bug Fixes

* fix license, bump ([ca66921](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/ca669215c8858e6e8d7e4e02bd527774be70b976))
* force CPU usage in testing ([474c6e0](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/474c6e012823dcd9fa50065ca4debc668503468d))
* formatting and linting ocntinue on error ([f014dc3](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/f014dc3622850cffd1edfd6789e10186b48bce88))
* release fix, bump ([596d371](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/596d371be69f55ddc28b5d0523f39ecb800b973d))
* resolve linting and formatting issues ([1666ac9](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/1666ac9a449c1a2553374d7c7f835af2194b721f))
* uv build ([b8269ef](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/b8269effb4dae19927b9ad4dbe174f51c1c18b43))

## [0.5.1](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.5.0...v0.5.1) (2025-06-23)


### Bug Fixes

* need fix for version control ([20cafbd](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/20cafbd019a80f947ebae005da6a1a9040d6d37e))

## [0.5.0](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.4.0...v0.5.0) (2025-06-23)


### Features

* add auto-installation to execute() function ([9a16c86](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/9a16c8639ca00c0937aff39514dbe51e8bd8a81c))
* improve code quality with logging, error handling, docs, and tests ([00cc7bc](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/00cc7bc2c94a1df0e2f9a061673610b435e6da1a))
* modernize packaging with pyproject.toml and uv ([90f5b44](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/90f5b4451341b0ad8f6cd138f81dd48118578baf))


### Bug Fixes

* add ipykernel to ui dependencies ([4d6a6e8](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/4d6a6e8db1f77a01e88da7d503c93b0418de85d0))
* correct uv PATH setup for Euler cluster ([a27a026](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/a27a026cc53dabceb2d8f30624fa4bec692c3cea))
* resolve conflict ([f0988d4](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/f0988d4096358e34db5fafebc973329b862acbea))
* resolve conflict ([ff5829b](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/ff5829bdb001d9156c2231ed2f8218dc514ea1ea))
* resolve conflict ([686bc4e](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/686bc4e9cef107f09c7be7c33d46be4a711c18df))
* resolve conflict ([c5cae1b](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/c5cae1b27a20db1a628827d20d9ca5af8278fb18))
* restrict Python version to <3.13 for dependency compatibility ([49603a0](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/49603a06dac2b9152f24bf7b078e4c42a9322481))
* separate CLI and programmatic interfaces ([6cd93c1](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/6cd93c11b824af6515f457761f234b7f2a584df6))
* update app.py to use transcribe_core ([d3b1f21](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/d3b1f21d7a3854679b470781f356278d947347ad))

## [0.4.0](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.3.3...v0.4.0) (2025-06-23)


### Features

* added app UI ([1a91972](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/1a91972b0dba8e161f72559fff6ce0195afb5b8e))


### Bug Fixes

* remove unnecessary newlines in app.py ([3a7d4b8](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/3a7d4b89ad308c7289131bac0326a1726e029d3b))

## [0.3.3](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.3.2...v0.3.3) (2025-06-20)


### Bug Fixes

* bump release ([381b1d2](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/381b1d2c7dff371352f7869eb38da478731f5e8f))

## [0.3.2](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.3.1...v0.3.2) (2025-06-19)


### Bug Fixes

* python3.9 missing requirements ([1ff44c2](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/1ff44c2bb3a9e29fb123dd7d417cb7042c23aed9))

## [0.3.1](https://github.com/Global-Health-Engineering/ghe_transcribe/compare/v0.3.0...v0.3.1) (2025-06-19)


### Bug Fixes

* added dependencies installation ([0bd6205](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/0bd62053e462c7fe85e06f57f0f2adcb28edc399))
* fixed typer interaction with if statement ([4b00bc4](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/4b00bc4573785fdb8c0dc9074335b0a22a85d43c))
* typo ([995de43](https://github.com/Global-Health-Engineering/ghe_transcribe/commit/995de4304088c98a6d04742b28c6a8c32a6a88d6))

