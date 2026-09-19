# OSM-derived inputs

The simulation runs without this directory's data by using station-level
walking times. Coordinate movement and density maps are enabled only when the
following files are supplied:

- `walk_times_osm.json`: `{station_name: minutes}`
- `walk_paths_osm.json`: direct station-to-gate coordinate sequences
- `event_zones_osm.json`: event-zone coordinates
- `walk_paths_event_osm.json`: station-to-zone and zone-to-gate paths
- `walk_times_event_osm.json`: travel times for those event paths

The `*.example.json` files document the accepted schemas. Their coordinates
are deliberately schematic and are **not** observations or validated routes.
Do not rename them to the production filenames for research runs. Generate
the production files from a documented OSM snapshot and record its date,
projection, routing profile, and licence attribution.

Coordinates can be longitude/latitude pairs or projected metre coordinates.
All files must use the same coordinate reference system. Longitude/latitude
route lengths are calculated with a local equirectangular approximation;
projected coordinates are treated as metres.
