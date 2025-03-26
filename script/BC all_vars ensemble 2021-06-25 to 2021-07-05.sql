EXPORT DATA OPTIONS (
  uri='gs://gencast/weathernext/weathernext_bc_heatwave_2021_avg/*.csv',
  format='CSV',
  overwrite=true
)
AS
SELECT
  t1.init_time,
  forecast.time AS forecast_time,
  ST_ASTEXT(t1.geography_polygon) AS geography_polygon_text,
  ST_ASTEXT(t1.geography) AS geography_text,
  
  AVG(ensemble.total_precipitation_12hr)     AS avg_total_precipitation_12hr,
  AVG(ensemble.`100m_u_component_of_wind`)     AS avg_100m_u_component_of_wind,
  AVG(ensemble.`100m_v_component_of_wind`)     AS avg_100m_v_component_of_wind,
  AVG(ensemble.`10m_u_component_of_wind`)      AS avg_10m_u_component_of_wind,
  AVG(ensemble.`10m_v_component_of_wind`)      AS avg_10m_v_component_of_wind,
  AVG(ensemble.`2m_temperature`)               AS avg_2m_temperature,
  AVG(ensemble.mean_sea_level_pressure)        AS avg_mean_sea_level_pressure,
  AVG(ensemble.sea_surface_temperature)        AS avg_sea_surface_temperature
FROM
  `glossy-apex-447402-k5.weathernext_gen_forecasts.126478713_1_0` AS t1,
  UNNEST(t1.forecast) AS forecast,
  UNNEST(forecast.ensemble) AS ensemble
WHERE
  -- British Columbia bounding box
  ST_INTERSECTS(
    t1.geography_polygon,
    ST_GEOGFROMTEXT('POLYGON((-139 48, -113 48, -113 60, -139 60, -139 48))')
  )
  -- Use init_time between June 11 and June 21, 2021 (14 days before forecast start)
  AND t1.init_time BETWEEN TIMESTAMP('2021-06-11 00:00:00 UTC') AND TIMESTAMP('2021-06-21 23:59:59 UTC')
  -- Use forecast times between June 25, 2021 and July 05, 2021
  AND forecast.time BETWEEN TIMESTAMP('2021-06-25 00:00:00 UTC') AND TIMESTAMP('2021-07-05 23:59:59 UTC')
  -- Only include forecasts with lead times between 0 and 14 days
  AND DATETIME_DIFF(forecast.time, t1.init_time, DAY) BETWEEN 0 AND 14
GROUP BY
  t1.init_time,
  forecast.time,
  ST_ASTEXT(t1.geography_polygon),
  ST_ASTEXT(t1.geography)
ORDER BY
  forecast.time,
  t1.init_time;
