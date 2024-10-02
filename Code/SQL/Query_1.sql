/* SELECT 
	COUNT(request_id) AS "Total Request", 
	SUM(n_cans) AS "Total Cans",
	SUM(n_glassbottles) AS "Total Glass Bottles",
	SUM(n_plasticbottles) AS "Total Plastic Bottles"
FROM requests; */

SELECT date_image
FROM requests
WHERE 

