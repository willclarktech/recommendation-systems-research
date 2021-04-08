# Recommendation Systems Research

Research into recommendation systems.

## Troubleshooting

### Poetry

I had trouble installing `gym` with Poetry because of its `Pillow` dependency and something to do with `zlib`. Setting `PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"` fixed this problem.
