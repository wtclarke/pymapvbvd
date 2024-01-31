This document contains the pymapvbvd release history in reverse chronological order.

0.5.7 (Wednesday 31st January 2024)
-----------------------------------
- Added flag to disable line reflection. Thanks to FrankZijlstra for contributing.
- Fixed depreciation warning (Issue 42). Thanks to MusicInMyBrain for reporting. 

0.5.6 (Wednesday 11th October 2023)
-----------------------------------
- Fixed issue with large files on Windows. Thanks to FrankZijlstra for reporting.
- Fixed subtle bug with precisely sized files interacting with memory chunking size. Thanks to FrankZijlstra for reporting. 

0.5.5 (Tuesday 10th October 2023)
---------------------------------
- Suppress warning `RuntimeWarning: invalid value encountered in cast`.
- Python 3.7 is no longer supported.

0.5.4 (Monday 10th July 2023)
-----------------------------
- Fix issue intorduced in `0.5.3` where tests were vendored as a top level package.
- This changelog is now included in the sdist.

0.5.3 (Tuesday 7th July 2023)
-----------------------------
- Performance enhancements and error checking for corrupted files. With thanks to Alex Craven
- Removed unnecessary build and test requirements from the `requirements.yml` file.

0.5.2 (Tuesday 10th January 2023)
---------------------------------
- Updated build dependencies (for pyproject.toml build)

0.5.1 (Tuesday 10th January 2023)
---------------------------------
- Move to pyproject.toml build

0.5.0 (Tuesday 10th January 2023)
---------------------------------
- Enable support for Python 3.11
- Remove support for Python 3.6
- AUtomatic Pypi upload on publication

0.4.8 (Thursday 27th January 2022)
----------------------------------
- Release to enable Zenodo automatic archiving 
