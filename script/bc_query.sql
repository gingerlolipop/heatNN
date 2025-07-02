-- bc_query.sql
-- This query retrieves 2m temperature forecasts for British Columbia (with some buffer area)
-- The polygon used roughly covers British Columbia with extra space for a buffer.
-- request from https://console.cloud.google.com/bigquery/analytics-hub/discovery/projects/gcp-public-data-weathernext/locations/us/dataExchanges/weathernext_19397e1bcb7/listings/weathernext_gen_forecasts_193b7e476d7?pli=1&authuser=1&project=glossy-apex-447402-k5
-- The query retrieves the average 2m temperature for each forecast time for the specified time range.
-- The query uses the UNNEST function to flatten the forecast and ensemble arrays.
-- The query groups the results by the initial time and forecast time.
-- The query orders the results by the initial time and forecast time.

SELECT
  t1.init_time, -- Initial time
  forecast.time AS forecast_time, -- Forecast time
  AVG(ensemble.`2m_temperature`) -- Average 2m temperature
FROM
  `dataset` AS t1, -- Replace `dataset` with the dataset ID
  UNNEST(forecast) AS forecast, -- Flatten the forecast array
  UNNEST(ensemble) AS ensemble -- Flatten the ensemble array
WHERE
  ST_CONTAINS(
    t1.geography_polygon, 
    ST_GEOGFROMTEXT('POLYGON((-139 48, -139 60, -120 60, -120 48, -139 48))')
  )  -- British Columbia with buffer
  AND t1.init_time BETWEEN TIMESTAMP('2024-12-10 00:00:00 UTC')
                       AND TIMESTAMP('2024-12-12 00:00:00 UTC')
GROUP BY
  1, 2 -- Group by initial time and forecast time
ORDER BY
  t1.init_time, forecast_time; -- Order by initial time and forecast time
