# Query 2021-06-25 to 2021-06-28 t2m data from WeatherNext
SELECT
  t1.init_time,
  forecast.time AS forecast_time,
  AVG(ensemble.`2m_temperature`) AS avg_2m_temperature,
  ST_ASTEXT(t1.geography_polygon) AS geography_polygon_text,
  ST_ASTEXT(t1.geography) AS geography_text
FROM
  `glossy-apex-447402-k5`.`weathernext_gen_forecasts`.`126478713_1_0` AS t1,
  UNNEST(t1.forecast) AS forecast,
  UNNEST(forecast.ensemble) AS ensemble
WHERE
  ST_INTERSECTS(t1.geography_polygon, ST_GEOGFROMTEXT('POLYGON((-124.0 48.8, -122.5 48.8, -122.5 49.8, -124.0 49.8, -124.0 48.8))'))
  AND t1.init_time BETWEEN TIMESTAMP('2021-06-25 00:00:00 UTC')
  AND TIMESTAMP('2021-06-28 00:00:00 UTC')
GROUP BY
  t1.init_time,
  forecast.time,
  ST_ASTEXT(t1.geography_polygon),
  ST_ASTEXT(t1.geography)
ORDER BY
  t1.init_time,
  forecast.time;
