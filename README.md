# Dibya's Blog

### How to use

Install Hugo and Quarto

```bash
wget -qO- https://github.com/gohugoio/hugo/releases/download/v0.147.9/hugo_0.147.9_Linux-64bit.tar.gz | gunzip | tar xvf - hugo -C $HOME/.local/bin
# Or uvx --from quarto-cli quarto, or just pip install quarto-cli
uv tool install quarto-cli 
```

```bash
# or you can install quarto (uv tool install quarto-cli)
quarto render content --to hugo-md
hugo server -D
```



### 07/04/2025

At long last, I've updated my blog off of Jekyll (after a few years of inactivity, I realized that trying to remember how to compile my blog with Ruby was a lost cause).

The workflow is simpler than what I had before:

1. Write in Jupyter Notebook (metadata at the top of each notebook)
2. Quarto converts the notebook to markdown
3. Hugo converts the markdowns to a site.


I've tried to make it as minimal as possible.