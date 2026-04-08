publish:
	quarto publish gh-pages ./content --no-render

clean:
	rm -rf content/_freeze
	rm -rf content/_size
	rm -rf content/.quarto
	rm -rf content/_site
	rm -rf content/site_libs

# restore:
# 	git restore --staged --worktree content/ old/
# 	git clean -fd content/ old/