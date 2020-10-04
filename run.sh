sudo apt install ruby-dev ruby-bundler nodejs

bundle clean --force

bundle install

bundle exec jekyll liveserve
