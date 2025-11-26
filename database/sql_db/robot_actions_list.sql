/*M!999999\- enable the sandbox mode */ 
-- MariaDB dump 10.19  Distrib 10.6.22-MariaDB, for debian-linux-gnu (x86_64)
--
-- Host: localhost    Database: ARTO
-- ------------------------------------------------------
-- Server version	10.6.22-MariaDB-0ubuntu0.22.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `robot_actions_list`
--

DROP TABLE IF EXISTS `robot_actions_list`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robot_actions_list` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `tcp_frame` varchar(255) DEFAULT NULL COMMENT 'pose of the end effector expressed in the origin frame',
  `tcp_wrench` varchar(255) DEFAULT NULL COMMENT 'force applied at the end effector expressed in the origin frame',
  `tool_id` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `robot_actions_list`
--

LOCK TABLES `robot_actions_list` WRITE;
/*!40000 ALTER TABLE `robot_actions_list` DISABLE KEYS */;
INSERT INTO `robot_actions_list` VALUES (1,'{ -0.191281,0.717774,-0.0554585,0.99557,-0.0118746,-0.0843926,-0.0397035 }',NULL,'FingerTool'),(2,'{ -0.195829,0.719161,-0.0950989,0.995356,-0.0100045,-0.0877808,-0.0382305 }','{ -0.869906 , 0.389311 , -4.90833 , 0 , 0 , -0 }','FingerTool'),(3,'{ -0.152553,0.610067,0.12135,0.995657,-0.0304671,0.0752546,-0.0455643 }',NULL,'gripper_hande'),(4,'{ -0.122708,0.598853,-0.0542896,0.990503,-0.0178988,0.133906,-0.0255314 }','{ -9.38013 , -1.19738 , 26.4307 , 0 , 0 , 0 }','gripper_hande'),(5,'{ -0.122708,0.598853,-0.0542896,0.990503,-0.0178988,0.133906,-0.0255314 }','{ -36.8904 , -0.104684 , 20.9544 , 0 , 0 , 0 }','gripper_hande'),(6,'{ -0.122708,0.598853,-0.0542896,0.990503,-0.0178988,0.133906,-0.0255314 }','{ -3.75275 , 0.394615 , -5.17311 , 0 , 0 , 0 }','gripper_hande'),(7,'{ -0.0905275,0.601499,-0.0491832,0.998166,-0.046975,-0.00832729,-0.0372527 }','{ -1.62342 , -1.91792 , 27.9586 , 0 , 0 , -0 }','gripper_hande'),(8,'{ -0.0905275,0.601499,-0.0491832,0.998166,-0.046975,-0.00832729,-0.0372527 }','{ -29.4697 , 0.540184 , 30.5163 , 0 , 0 , -0 }','gripper_hande'),(9,'{ -0.0905275,0.601499,-0.0491832,0.998166,-0.046975,-0.00832729,-0.0372527 }','{ -5.02974 , 0.766392 , -3.88772 , 0 , 0 , -0 }','gripper_hande'),(10,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ 4.35974 , -1.69725 , 27.6787 , 0 , 0 , -0 }','gripper_hande'),(11,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ -22.3537 , 0.346358 , 36.0582 , 0 , 0 , -0 }','gripper_hande'),(12,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ -5.75061 , 0.65323 , -2.73929 , 0 , 0 , -0 }','gripper_hande'),(13,'{ -0.0535719,0.604978,0.084152,0.994756,0.00125754,-0.0760661,-0.068357 }',NULL,'gripper_hande'),(14,'{ -0.0355985,0.604052,-0.0645985,0.978332,-0.0296377,-0.203658,-0.0226209 }','{ 9.28961 , -1.47978 , 26.4483 , 0 , 0 , -0 }','gripper_hande'),(15,'{ -0.0355985,0.604052,-0.0645985,0.978332,-0.0296377,-0.203658,-0.0226209 }','{ -15.5442 , -0.226689 , 39.4756 , 0 , 0 , -0 }','gripper_hande'),(16,'{ -0.0355985,0.604052,-0.0645985,0.978332,-0.0296377,-0.203658,-0.0226209 }','{ -6.16505 , 0.469219 , -1.66494 , 0 , 0 , -0 }','gripper_hande'),(17,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ 8.24024 , -2.0081 , 26.7594 , 0 , 0 , -0 }','gripper_hande'),(18,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ 36.5287 , -4.51488 , 25.1847 , 0 , 0 , -0 }','gripper_hande'),(19,'{ -0.0608036,0.60237,-0.0553071,0.992095,-0.0424517,-0.114613,-0.0284429 }','{ 3.95062 , -0.123893 , -5.03759 , 0 , 0 , -0 }','gripper_hande'),(20,'{ -0.122708,0.598853,-0.0542896,0.990503,-0.0178988,0.133906,-0.0255314 }','{ -5.52614 , -1.36656 , 27.4881 , 0 , 0 , 0 }','gripper_hande'),(21,'{ -0.122708,0.598853,-0.0542896,0.990503,-0.0178988,0.133906,-0.0255314 }','{ 5.88222 , -0.0283373 , -2.52956 , 0 , 0 , 0 }','gripper_hande'),(22,' { -0.425755,0.609789,0.507875,-0.492916,-0.522329,0.451328,0.529631 }',NULL,'FingerTool'),(23,'{ -0.562113,0.633953,0.475496,0.56643,0.574696,-0.424028,-0.411195 }','{ -4.76494 , -0.107745 , -1.51119 , 0 , 0 , -0 }','FingerTool'),(24,'{ -0.329657,0.413286,0.213528,0.695008,0.696304,-0.0558964,-0.170298 }',NULL,'FingerTool'),(25,'{ -0.488061,0.370539,-0.0412383,0.719777,0.684245,-0.0444769,-0.108403 }','{ -1.06187 , 0.475926 , -4.86271 , 0 , 0 , 0 }','FingerTool');
/*!40000 ALTER TABLE `robot_actions_list` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-11-11 14:19:08
