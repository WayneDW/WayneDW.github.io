sudo apt install ruby-dev ruby-bundler nodejs
sleep 1s
bundle clean --force
sleep 1s
bundle install
sleep 1s
bundle exec jekyll liveserve
