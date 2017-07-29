# EyePrint
Create an eye sketch using a dynamic programing algorithm.

1. Given an image of an eye, convert its data to a 3d representation:
   X and Y axis are the coordinates and Z is the color data
   
2. Then the eye image is recursively divided into sectors to find the optimal eye partition.
   each sector is rotated parallel to X axis.
   
3. Each sector is then recursively divided to find optimal number of segments and their position.

4. For each sector a plane that minimizes the distances of each (X,Y,Z) point is computed.
   
5. A dynamic programing algorithm is used in order to calculate the optimal partition to k segments.
   
6. In order to find the ideal number of partitions for the whole eye and for each sector (slice)
   a binary search is used to find the "elbow" point,
   the point that is farthest from the line that connects (1,f(1)) with (n,f(n))=(n,0),
   where (1,f(1)) is the sum of distances from the ideal plane of partition to 1 segment,
   and (n,f(n)) is the sum of distances from the ideal plane of partition to n segments.
