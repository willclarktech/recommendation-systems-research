#!/usr/bin/env sh
docker run -t --rm -p 5000:5000 -e PORT=5000 -e DATABASE_URL="sqlite:///nodedatabase.db" openmined/grid-domain
