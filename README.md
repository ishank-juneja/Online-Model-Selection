## Legend

1. Project specific `.py` files go into `src` or sub-dir of `src`. All plotting scripts go in `src/plotting`. Avoid any `matplotlib` or related imports in main `src` files
2. All scripts (think `.sh` files, and template `.py` files) that write to `results` should be in `scripts/`
2. All `.ipynb` files go into `notebooks`. Using google colab is preffered but files should be downloaded into this folder once done
3. All figures/videos/plots go in the corresponding sub-dir of `results`. Example all `.png` files in `results/figures`
4. All models (bullet/mujoco physcis based, trained, or otherwise) go into `models` or sub-dir of `models`
5. Any documentation that does not belong in the README.md (text files for minor references etc.) and things like referred PDFs go into `notes`
6. All local datasets and problem instance files go into `data` or sub-dir of `data`

## Append project .gitignore with

```
notes/
data/
```

## Command to be able to add an empty directory tree to git
Command used to push this empty template repo to git/guthub was. Run this before pushing whenever a new dir is added to this template

`find . -type d -empty -not -path "./.git/*" -exec touch {}/.gitkeep \;`

Once cloned, run the following to get rid of these `.gitkeep` files and keep your project pristine

`find . -name '*.gitkeep' -delete`

## Why this exists
When a python-project reaches a stage where it is ready to transition from a collection of scripts to a "code-base" then at that point (this should happen once the viability of the project has been established) a remote (typically a GitHub) should be created for it using this template to save time and maintain standardization in relative paths.
