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
-- Table structure for table `aural_atomics`
--

DROP TABLE IF EXISTS `aural_atomics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `aural_atomics` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` varchar(255) NOT NULL,
  `atomic_type` varchar(32) NOT NULL DEFAULT 'matchTemplate',
  `max_recording_length` int(11) NOT NULL DEFAULT 60 COMMENT 'The maximum time in seconds for which to listen',
  `times_to_recognize` int(11) NOT NULL DEFAULT 3 COMMENT 'The minimum number of times with which to recognize a template',
  `path_to_template` varchar(255) NOT NULL COMMENT '/assetName/TemplateName.wav',
  PRIMARY KEY (`id`),
  UNIQUE KEY `aural_atomics_unique` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `aural_results_temp`
--

DROP TABLE IF EXISTS `aural_results_temp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `aural_results_temp` (
  `atomic_name` varchar(255) NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `recorded_audio` longblob DEFAULT NULL COMMENT 'audio used for the match in flac format',
  PRIMARY KEY (`atomic_name`,`timestamp`),
  CONSTRAINT `aural_results_temp_aural_atomics_FK` FOREIGN KEY (`atomic_name`) REFERENCES `aural_atomics` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `aurals_with_microphone`
--

DROP TABLE IF EXISTS `aurals_with_microphone`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `aurals_with_microphone` (
  `atomic_id` int(11) NOT NULL,
  `microphone_id` int(11) NOT NULL,
  PRIMARY KEY (`atomic_id`,`microphone_id`),
  KEY `micro_id_idx` (`microphone_id`),
  CONSTRAINT `fk_athomic_id` FOREIGN KEY (`atomic_id`) REFERENCES `aural_atomics` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_micro_id` FOREIGN KEY (`microphone_id`) REFERENCES `microphones` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `camera_wrt_robot`
--

DROP TABLE IF EXISTS `camera_wrt_robot`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `camera_wrt_robot` (
  `robot_id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  `wrt_base_or_flange` tinyint(4) NOT NULL,
  `offset_cam_rob` double DEFAULT NULL,
  PRIMARY KEY (`robot_id`,`camera_id`),
  KEY `fk_cam_2` (`camera_id`),
  CONSTRAINT `fk_cam_1` FOREIGN KEY (`robot_id`) REFERENCES `robotic_arms` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_cam_2` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cameras`
--

DROP TABLE IF EXISTS `cameras`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cameras` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `version` double NOT NULL,
  `ip_address` varchar(32) DEFAULT NULL,
  `type` varchar(16) DEFAULT NULL,
  `driver` varchar(255) DEFAULT NULL,
  `cockpit_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `cameras_cockpits_FK` (`cockpit_id`),
  CONSTRAINT `cameras_cockpits_FK` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cameras_in_procedure`
--

DROP TABLE IF EXISTS `cameras_in_procedure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cameras_in_procedure` (
  `procedure_id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  PRIMARY KEY (`procedure_id`,`camera_id`),
  KEY `fk_cameras_in_procedure_1_idx` (`camera_id`),
  CONSTRAINT `fk_cameras_in_procedure_1` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_cameras_in_procedure_2` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cockpits`
--

DROP TABLE IF EXISTS `cockpits`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cockpits` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `panels_folder_name` varchar(255) DEFAULT NULL COMMENT 'used in panel detection network',
  `max_robot_speed` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `component_moves_by`
--

DROP TABLE IF EXISTS `component_moves_by`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `component_moves_by` (
  `component_id` int(11) NOT NULL,
  `tool_id` int(11) NOT NULL,
  PRIMARY KEY (`component_id`,`tool_id`),
  KEY `fk_comp_2` (`tool_id`),
  CONSTRAINT `fk_comp_1` FOREIGN KEY (`component_id`) REFERENCES `features` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_comp_2` FOREIGN KEY (`tool_id`) REFERENCES `tools` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `components_state`
--

DROP TABLE IF EXISTS `components_state`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `components_state` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `od_class` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cv_alternation_atomic`
--

DROP TABLE IF EXISTS `cv_alternation_atomic`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cv_alternation_atomic` (
  `id` int(11) NOT NULL,
  `alternation_atomic_name` varchar(255) DEFAULT NULL,
  `alternated_atomic_name` varchar(255) DEFAULT NULL,
  `order_index` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `cv_alternation_atomic_cv_atomics_FK` (`alternation_atomic_name`),
  KEY `cv_alternation_atomic_cv_atomics_FK_1` (`alternated_atomic_name`),
  CONSTRAINT `cv_alternation_atomic_cv_atomics_FK` FOREIGN KEY (`alternation_atomic_name`) REFERENCES `cv_atomics` (`name`) ON UPDATE CASCADE,
  CONSTRAINT `cv_alternation_atomic_cv_atomics_FK_1` FOREIGN KEY (`alternated_atomic_name`) REFERENCES `cv_atomics` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cv_atomics`
--

DROP TABLE IF EXISTS `cv_atomics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cv_atomics` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` varchar(512) NOT NULL,
  `atomic_type` varchar(32) NOT NULL,
  `rotation_direction` tinyint(1) DEFAULT NULL,
  `rotation_quantity` int(11) DEFAULT NULL,
  `crop_proportion` text DEFAULT NULL,
  `feature_to_crop` varchar(255) DEFAULT NULL,
  `text_color` text DEFAULT NULL,
  `background_color` text DEFAULT NULL,
  `ocr_string_to_read` varchar(255) DEFAULT NULL,
  `ocr_int_to_read` int(11) DEFAULT NULL,
  `od_class` int(11) DEFAULT NULL,
  `flashing` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `cv_atomics_unique` (`name`),
  KEY `cv_atomics_features_FK` (`feature_to_crop`),
  CONSTRAINT `cv_atomics_features_FK` FOREIGN KEY (`feature_to_crop`) REFERENCES `features` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cv_atomics_allowed_on`
--

DROP TABLE IF EXISTS `cv_atomics_allowed_on`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cv_atomics_allowed_on` (
  `atomic_id` int(11) NOT NULL,
  `component_id` int(11) NOT NULL,
  PRIMARY KEY (`atomic_id`,`component_id`),
  KEY `fk_cv_2` (`component_id`),
  CONSTRAINT `fk_cv_1` FOREIGN KEY (`atomic_id`) REFERENCES `cv_atomics` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_cv_2` FOREIGN KEY (`component_id`) REFERENCES `features` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cv_atomics_allowed_with`
--

DROP TABLE IF EXISTS `cv_atomics_allowed_with`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cv_atomics_allowed_with` (
  `atomic_id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  PRIMARY KEY (`atomic_id`,`camera_id`),
  KEY `fk_cv_4` (`camera_id`),
  CONSTRAINT `fk_cv_3` FOREIGN KEY (`atomic_id`) REFERENCES `cv_atomics` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_cv_4` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cv_results_temp`
--

DROP TABLE IF EXISTS `cv_results_temp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `cv_results_temp` (
  `atomic_name` varchar(255) NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `frames_list` longtext DEFAULT NULL COMMENT 'save as a list of array',
  PRIMARY KEY (`atomic_name`,`timestamp`),
  CONSTRAINT `cv_results_temp_cv_atomics_FK` FOREIGN KEY (`atomic_name`) REFERENCES `cv_atomics` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `executions`
--

DROP TABLE IF EXISTS `executions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `executions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `start_timestamp` bigint(20) NOT NULL,
  `end_timestamp` bigint(20) DEFAULT NULL,
  `output` varchar(32) DEFAULT NULL,
  `report_file_path` varchar(255) DEFAULT NULL,
  `diagnostic_file_path` varchar(255) DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id_idx` (`user_id`),
  CONSTRAINT `user_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `features`
--

DROP TABLE IF EXISTS `features`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `features` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `cv_map_on_panel` text DEFAULT NULL COMMENT 'vector of 4 coordinates in pixels on reference panel image, used in cv to identify the component',
  `3d_offset_to_panel` text DEFAULT NULL COMMENT 'component offset related to the position of the panel, identified by qr code. It is a quaternion',
  `feature_type` varchar(32) NOT NULL COMMENT 'component, text or QR code',
  `component_type` varchar(32) DEFAULT NULL COMMENT 'button, screen, analogic display, knob or lever',
  `panel_belonging` varchar(64) NOT NULL,
  `component_current_state` int(11) DEFAULT NULL COMMENT 'visual state distinguishable by OD classes',
  `component_actual_value` varchar(64) DEFAULT NULL COMMENT 'stringa leggibile con OCR, spesso è il valore di una feature text e va riempito il campo solo nella feature text sul display, non sul componente fisico da muovere',
  `text` varchar(255) DEFAULT NULL,
  `text_map_on_display` text DEFAULT NULL COMMENT '[x1, y1, x2, y2] pixels position in analogic display of belonging',
  `number_of_texts_on_display` int(11) DEFAULT NULL COMMENT 'numero di feature text presenti su un display analogico',
  `color` varchar(32) DEFAULT NULL,
  `lever_diameter` int(11) DEFAULT NULL,
  `conversion_lookup_table_to_display` text DEFAULT NULL COMMENT '"[list of angles (x)]; [list of values (y)]" da mettere sulla feature text',
  `selection_vector_lookup_table` text DEFAULT NULL COMMENT '{[bool], [bool], [bool], [bool], [bool], [bool]} da mettere sulla feature text',
  `reference_component_id` int(11) DEFAULT NULL COMMENT 'da mettere nella feature che possiede current_value per specificare qual è la feature id che permette il cambiamento di questo valore',
  PRIMARY KEY (`id`),
  UNIQUE KEY `features_unique` (`name`),
  KEY `fk_feature_1_idx` (`component_current_state`),
  KEY `fk_feature_2` (`panel_belonging`),
  CONSTRAINT `features_suts_FK` FOREIGN KEY (`panel_belonging`) REFERENCES `panels` (`name`) ON UPDATE CASCADE,
  CONSTRAINT `fk_feature_1` FOREIGN KEY (`component_current_state`) REFERENCES `components_state` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `look_at_features_with`
--

DROP TABLE IF EXISTS `look_at_features_with`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `look_at_features_with` (
  `camera_id` int(11) NOT NULL,
  `feature_id` int(11) NOT NULL,
  PRIMARY KEY (`camera_id`,`feature_id`),
  KEY `fk_look_2` (`feature_id`),
  CONSTRAINT `fk_look_1` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_look_2` FOREIGN KEY (`feature_id`) REFERENCES `features` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `microphones`
--

DROP TABLE IF EXISTS `microphones`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `microphones` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL,
  `version` double NOT NULL DEFAULT 0,
  `config` varchar(255) DEFAULT NULL,
  `cockpit_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `micro_cock_idx` (`cockpit_id`),
  CONSTRAINT `micro_cock` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `microphones_in_procedure`
--

DROP TABLE IF EXISTS `microphones_in_procedure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `microphones_in_procedure` (
  `procedure_id` int(11) NOT NULL,
  `microphone_id` int(11) NOT NULL,
  PRIMARY KEY (`procedure_id`,`microphone_id`),
  KEY `fk_microphone_in_procedure_1_idx` (`microphone_id`),
  CONSTRAINT `fk_microphone_in_procedure_1` FOREIGN KEY (`microphone_id`) REFERENCES `microphones` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_microphone_in_procedure_2` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `panels`
--

DROP TABLE IF EXISTS `panels`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `panels` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `path_to_image` varchar(255) DEFAULT NULL COMMENT 'used during panels detection',
  `cockpit_position` text DEFAULT NULL COMMENT 'helps map elements belonging to the same cockpit, for cv',
  `coordinates_wrt_robot` text DEFAULT NULL,
  `cockpit_belonging` int(11) NOT NULL,
  `total_buttons` int(11) NOT NULL DEFAULT 0,
  `total_screens` int(11) NOT NULL DEFAULT 0,
  `total_analogic_displays` int(11) NOT NULL DEFAULT 0,
  `total_levers` int(11) NOT NULL DEFAULT 0,
  `total_knobs` int(11) NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  UNIQUE KEY `suts_unique` (`name`),
  KEY `fk_sut_1_idx` (`cockpit_belonging`),
  CONSTRAINT `fk_sut_1` FOREIGN KEY (`cockpit_belonging`) REFERENCES `cockpits` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `possible_states_for_component`
--

DROP TABLE IF EXISTS `possible_states_for_component`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `possible_states_for_component` (
  `component_id` int(11) NOT NULL,
  `state_id` int(11) NOT NULL,
  PRIMARY KEY (`component_id`,`state_id`),
  KEY `fk_possible_states_for_component_1_idx` (`state_id`),
  CONSTRAINT `fk_states_1` FOREIGN KEY (`component_id`) REFERENCES `features` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_states_2` FOREIGN KEY (`state_id`) REFERENCES `components_state` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `procedures`
--

DROP TABLE IF EXISTS `procedures`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `procedures` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(128) NOT NULL,
  `json_file_path` varchar(255) NOT NULL,
  `user_creator` int(11) DEFAULT NULL,
  `cockpit_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`),
  KEY `creator_idx` (`user_creator`),
  KEY `fk_procedures_queue_created_1_idx` (`cockpit_id`),
  CONSTRAINT `creator` FOREIGN KEY (`user_creator`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_procedures_queue_created_1` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `procedures_queue`
--

DROP TABLE IF EXISTS `procedures_queue`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `procedures_queue` (
  `procedure_id` int(11) NOT NULL,
  `execution_id` int(11) NOT NULL,
  `order_index` int(11) NOT NULL,
  PRIMARY KEY (`procedure_id`,`execution_id`),
  KEY `procedures_queue_procedures_executions_FK` (`execution_id`),
  CONSTRAINT `procedures_queue_procedures_FK` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `procedures_queue_procedures_executions_FK` FOREIGN KEY (`execution_id`) REFERENCES `executions` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `robotic_arms`
--

DROP TABLE IF EXISTS `robotic_arms`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robotic_arms` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL,
  `version` double NOT NULL DEFAULT 0,
  `ip_address` varchar(32) DEFAULT NULL,
  `home_pose` text DEFAULT NULL,
  `current_pose` text DEFAULT NULL,
  `current_tool` int(11) DEFAULT NULL,
  `cockpit_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `cock_rob_idx` (`cockpit_id`),
  KEY `fk_robotic_arm_1` (`current_tool`),
  CONSTRAINT `cock_rob` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_robotic_arm_1` FOREIGN KEY (`current_tool`) REFERENCES `tools` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `robotic_atomics`
--

DROP TABLE IF EXISTS `robotic_atomics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robotic_atomics` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` varchar(512) NOT NULL,
  `atomic_type` varchar(32) NOT NULL,
  `gripper_values` text DEFAULT NULL COMMENT '{[m], [m/s], [N]}',
  `joints_values` text DEFAULT NULL COMMENT '{[rad], [rad], [rad], [rad], [rad], [rad]}',
  `offset_pose` text DEFAULT NULL COMMENT '{[double], [double], [double], [double], [double], [double]}',
  `force_selection_vector` text DEFAULT NULL COMMENT '{[bool], [bool], [bool], [bool], [bool], [bool]}',
  `force_direction_vector` text DEFAULT NULL COMMENT '{[N], [N], [N], N/m, N/m, N/m}',
  `force_limits_vector` text DEFAULT NULL COMMENT '{[double], [double], [double], [double], [double], [double]}',
  `force_type` int(11) DEFAULT NULL COMMENT '1, 2 or 3',
  `force_time` double DEFAULT NULL COMMENT 'sec',
  `threshold_target_value` text DEFAULT NULL COMMENT '{[N], [N], [N], N/m, N/m, N/m}',
  `threshold_selection_vector` text DEFAULT NULL COMMENT '{[bool], [bool], [bool], [bool], [bool], [bool]}',
  `component_target_value` varchar(64) DEFAULT NULL COMMENT 'the value to be achieved by the component through robotics',
  `task_frame` int(11) DEFAULT NULL COMMENT 'componente fisico di riferimento o quello da muovere',
  `reference_force_profile` text DEFAULT NULL COMMENT '{[N, N, N, N/m, N/m, N/m], [N, N, N, N/m, N/m, N/m], ...}',
  PRIMARY KEY (`id`),
  UNIQUE KEY `robotic_atomics_unique` (`name`),
  KEY `fk_atomic_1` (`task_frame`),
  CONSTRAINT `fk_atomic_1` FOREIGN KEY (`task_frame`) REFERENCES `features` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `robotic_atomics_allowed_on`
--

DROP TABLE IF EXISTS `robotic_atomics_allowed_on`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robotic_atomics_allowed_on` (
  `atomic_id` int(11) NOT NULL,
  `feature_id` int(11) NOT NULL,
  PRIMARY KEY (`atomic_id`,`feature_id`),
  KEY `fk_allow_2` (`feature_id`),
  CONSTRAINT `fk_allow_2` FOREIGN KEY (`feature_id`) REFERENCES `features` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `robotic_atomics_allowed_on_robotic_atomics_FK` FOREIGN KEY (`atomic_id`) REFERENCES `robotic_atomics` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `robotic_results_temp`
--

DROP TABLE IF EXISTS `robotic_results_temp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robotic_results_temp` (
  `atomic_name` varchar(255) NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `registered_force_profile` longtext DEFAULT NULL COMMENT '[[time, N, N, N, Nm, Nm, Nm], [...], ...]',
  PRIMARY KEY (`atomic_name`,`timestamp`),
  CONSTRAINT `robotic_results_temp_robotic_atomics_FK` FOREIGN KEY (`atomic_name`) REFERENCES `robotic_atomics` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `robots_in_procedure`
--

DROP TABLE IF EXISTS `robots_in_procedure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `robots_in_procedure` (
  `procedure_id` int(11) NOT NULL,
  `robot_id` int(11) NOT NULL,
  PRIMARY KEY (`procedure_id`,`robot_id`),
  KEY `fk_robots_in_procedure_2` (`robot_id`),
  CONSTRAINT `fk_robots_in_procedure_1` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_robots_in_procedure_2` FOREIGN KEY (`robot_id`) REFERENCES `robotic_arms` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `steps_reports`
--

DROP TABLE IF EXISTS `steps_reports`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `steps_reports` (
  `execution_id` int(11) NOT NULL,
  `procedure_id` int(11) NOT NULL,
  `step_name` varchar(100) NOT NULL COMMENT 'extracted from xml trees',
  `counter` int(11) NOT NULL,
  `outcome` tinyint(4) DEFAULT NULL COMMENT '1: Passed, 0: Failed',
  `diagnostic_message` varchar(255) DEFAULT NULL,
  `cv_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, {analized_frames, score}}, {...}, ...',
  `aural_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, recorded_track}, {...}, ...',
  `robotic_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, registered_force_profile}, {...}, ...',
  PRIMARY KEY (`execution_id`,`procedure_id`,`step_name`,`counter`),
  KEY `steps_reports_procedures_FK` (`procedure_id`),
  CONSTRAINT `steps_reports_executions_FK` FOREIGN KEY (`execution_id`) REFERENCES `executions` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `steps_reports_procedures_FK` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `steps_reports_temp`
--

DROP TABLE IF EXISTS `steps_reports_temp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `steps_reports_temp` (
  `step_name` varchar(255) NOT NULL,
  `counter` int(11) NOT NULL,
  `outcome` tinyint(4) DEFAULT NULL COMMENT '1: Passed, 0: Failed',
  `diagnostic_message` varchar(255) DEFAULT NULL,
  `cv_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, {analized_frames, score}}, {...}, ...',
  `aural_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, recorded_track}, {...}, ...',
  `robotic_results` text DEFAULT NULL COMMENT '{atomic_id, timestamp, registered_force_profile}, {...}, ...',
  PRIMARY KEY (`step_name`,`counter`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tools`
--

DROP TABLE IF EXISTS `tools`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `tools` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL,
  `version` double NOT NULL DEFAULT 0,
  `tcp` text DEFAULT NULL,
  `payload` double DEFAULT NULL,
  `cog` text DEFAULT NULL,
  `cockpit_id` int(11) DEFAULT NULL,
  `min_opening` double NOT NULL DEFAULT 0,
  `max_opening` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `tools_cockpits_FK` (`cockpit_id`),
  CONSTRAINT `tools_cockpits_FK` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tools_in_procedure`
--

DROP TABLE IF EXISTS `tools_in_procedure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `tools_in_procedure` (
  `procedure_id` int(11) NOT NULL,
  `tool_id` int(11) NOT NULL,
  PRIMARY KEY (`procedure_id`,`tool_id`),
  KEY `fk_tools_in_procedure_2` (`tool_id`),
  CONSTRAINT `fk_tools_in_procedure_1` FOREIGN KEY (`procedure_id`) REFERENCES `procedures` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_tools_in_procedure_2` FOREIGN KEY (`tool_id`) REFERENCES `tools` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user_cockpit`
--

DROP TABLE IF EXISTS `user_cockpit`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_cockpit` (
  `cockpit_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`cockpit_id`,`user_id`),
  KEY `work_user_idx` (`user_id`),
  CONSTRAINT `work_cockpit` FOREIGN KEY (`cockpit_id`) REFERENCES `cockpits` (`id`),
  CONSTRAINT `work_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(64) NOT NULL,
  `password` varchar(64) NOT NULL,
  `last_access` datetime DEFAULT NULL,
  `role` char(1) NOT NULL DEFAULT 'N',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping routines for database 'ARTO'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-07-30 11:27:57
