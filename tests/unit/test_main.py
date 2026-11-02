import main


def test_parser_includes_cityjson_workflows() -> None:
    parser = main.parser_setup()

    assert parser.parse_args([
        'terrain',
        'terrain.tif',
        'terrain.city.json',
    ]).command == 'terrain'
    assert parser.parse_args([
        'terrain-tiles',
        'tiles',
    ]).command == 'terrain-tiles'
    assert parser.parse_args([
        'rbc-cityjson-tiles',
        'tiles',
    ]).command == 'rbc-cityjson-tiles'
    assert parser.parse_args([
        'sat-cityjson',
        'tiles',
        'city.json',
    ]).command == 'sat-cityjson'
    assert parser.parse_args([
        'ndsm-dir',
        'tiles',
        '--cmap-dir',
        'ndsm-cmap',
        '--verbose',
    ]).command == 'ndsm-dir'
    assert parser.parse_args([
        'predict-tile',
        '--model-path',
        'model.pt',
        'tiles',
        'tile.tif',
    ]).command == 'predict-tile'
    assert parser.parse_args([
        'predict-tile',
        '--model-path',
        'model.pt',
        'tiles',
    ]).tile_name is None
