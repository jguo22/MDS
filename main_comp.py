import time
import argparse
import threading
from RobotHandler_Simple import RobotHandler
from profiler import Profiler
from connection.ComputerReceiver import ComputerReceiver
from InputProcessor import InputProcessor
from connection.FrameSaver import FrameSaver
from connection.frame_info import FrameInfo
from vision.zone_utils import visualize_xy_locations, getPolygonCenter, visualize_convex_hulls
from vision.relativeCoordinates import world_to_relative, world_to_pixel
from vision.pixelTo3D import H_TOP, H_BOTTOM
from colors import GREEN_ZONE, RED_ZONE, GOLDEN_ZONE, GREEN_ZONE_OPP, RED_ZONE_OPP, GOLDEN_ZONE_OPP
import config


def main():
    # ----------------- GET INPUTS -----------------
    parser = argparse.ArgumentParser(description="Computer Video Receiver")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT,
                        help=f"Video port (default: {config.VIDEO_PORT})")
    parser.add_argument(
        "--coord-port",
        type=int,
        default=config.COMMAND_PORT,
        help=f"Coordinate port (default: {config.COMMAND_PORT})")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable video display")
    parser.add_argument(
        "--picture-delay",
        type=float,
        default=None,
        help="Delay in seconds between saving pictures (default: don't save)")
    args = parser.parse_args()

    window_name_top = "Top Camera"
    window_name_bottom = "Bottom Camera"

    # ----------------- CREATE RECEIVER AND PROCESSORS -----------------
    computer_receiver = ComputerReceiver(
        args.host, args.video_port, args.coord_port)

    # Get robot commander for sending commands
    robot_commander = computer_receiver.commander

    robotHandler = RobotHandler(robot_commander)
    inputProcessor = InputProcessor(
        robot_commander, window_name_top, robotHandler)

    # Only create FrameSaver if picture-delay is specified
    frame_saver = None
    if args.picture_delay is not None:
        frame_saver = FrameSaver(args.picture_delay, "images")

    # Create main profiler for frame processing pipeline
    main_profiler = Profiler(False)

    def process(frame_info: FrameInfo) -> None:
        # Process frame using FrameInfo
        main_profiler.start_frame()
        robotHandler.handleFrame(frame_info)
        main_profiler.record("robotHandler")

        # Only save frame if frame_saver was initialized
        if frame_saver is not None:
            frame_saver.saveFrame(frame_info)
            main_profiler.record("saveFrame")

        # Overlay YOLO segmentation results on frames first
        if robotHandler.result_top is not None:
            if robotHandler.result_top.masks is not None and len(
                    robotHandler.result_top.masks) > 0:
                # Plot segmentation masks on top frame
                frame_info.frame_top = robotHandler.result_top.plot(
                    boxes=True,
                    masks=True,
                    conf=True,
                    line_width=2,
                    labels=True
                )
                # Overlay convex hulls in cyan
                frame_info.frame_top = visualize_convex_hulls(
                    frame_info.frame_top,
                    robotHandler.result_top,
                    color=(255, 255, 0),  # Cyan
                    thickness=2
                )

        if robotHandler.result_bottom is not None:
            if robotHandler.result_bottom.masks is not None and len(
                    robotHandler.result_bottom.masks) > 0:
                # Plot segmentation masks on bottom frame
                frame_info.frame_bottom = robotHandler.result_bottom.plot(
                    boxes=True,
                    masks=True,
                    conf=True,
                    line_width=2,
                    labels=True
                )
                # Overlay convex hulls in cyan
                frame_info.frame_bottom = visualize_convex_hulls(
                    frame_info.frame_bottom,
                    robotHandler.result_bottom,
                    color=(255, 255, 0),  # Cyan
                    thickness=2
                )

        main_profiler.record("segmentation_viz")

        # Now add custom visualizations on top of segmentations
        # Visualize can locations on frames
        if len(robotHandler.cans) > 0:
            # Convert world coordinates to camera-relative coordinates
            can_locations_relative = [
                world_to_relative(can, robotHandler.robot_pose)
                for can in robotHandler.cans
            ]

            # Visualize on top camera
            frame_info.frame_top = visualize_xy_locations(
                frame_info.frame_top,
                can_locations_relative,
                robotHandler.robot_pose,
                is_top=True,
                color=(0, 255, 255),  # Yellow for cans
                radius=8,
                labels=[f"C{i}" for i in range(len(robotHandler.cans))]
            )

            # Visualize on bottom camera
            frame_info.frame_bottom = visualize_xy_locations(
                frame_info.frame_bottom,
                can_locations_relative,
                robotHandler.robot_pose,
                is_top=False,
                color=(0, 255, 255),  # Yellow for cans
                radius=8,
                labels=[f"C{i}" for i in range(len(robotHandler.cans))]
            )

        # Visualize zone polygons
        zone_colors = {
            GREEN_ZONE: (0, 255, 0),      # Green
            RED_ZONE: (0, 0, 255),        # Red
            GOLDEN_ZONE: (0, 215, 255),   # Gold
            GREEN_ZONE_OPP: (100, 200, 100),   # Light green
            RED_ZONE_OPP: (100, 100, 200),     # Light red
            GOLDEN_ZONE_OPP: (100, 215, 255)   # Light gold
        }

        # for zone_idx, zone in enumerate(robotHandler.zones):
        #     if zone is not None:
        #         # Get zone color
        #         color = zone_colors.get(
        #             zone_idx, (255, 255, 255))  # White default
        #
        #         # Reshape zone to (N, 2)
        #         if zone.ndim == 3:
        #             corners = zone.reshape(-1, 2)
        #         else:
        #             corners = zone if zone.ndim == 2 else zone.reshape(-1, 2)
        #
        #         # Draw polygon on top camera
        #         polygon_points_top = []
        #         for corner in corners:
        #             pixel = world_to_pixel(
        #                 (float(
        #                     corner[0]), float(
        #                     corner[1])), H_TOP)
        #             if pixel is not None:
        #                 polygon_points_top.append(pixel)
        #
        #         if len(polygon_points_top) >= 3:
        #             pts = np.array(polygon_points_top, dtype=np.int32)
        #             cv2.polylines(
        #                 frame_info.frame_top,
        #                 [pts],
        #                 isClosed=True,
        #                 color=color,
        #                 thickness=2)
        #
        #         # Draw polygon on bottom camera
        #         polygon_points_bottom = []
        #         for corner in corners:
        #             pixel = world_to_pixel(
        #                 (float(
        #                     corner[0]), float(
        #                     corner[1])), H_BOTTOM)
        #             if pixel is not None:
        #                 polygon_points_bottom.append(pixel)
        #
        #         if len(polygon_points_bottom) >= 3:
        #             pts = np.array(polygon_points_bottom, dtype=np.int32)
        #             cv2.polylines(
        #                 frame_info.frame_bottom,
        #                 [pts],
        #                 isClosed=True,
        #                 color=color,
        #                 thickness=2)
        #
        #         # Get zone center
        #         center_x, center_y = getPolygonCenter(zone)
        #         center_relative = world_to_relative(
        #             (center_x, center_y), robotHandler.robot_pose)
        #
        #         # Convert corners to camera-relative coordinates
        #         corners_relative = []
        #         for corner in corners:
        #             rel = world_to_relative(
        #                 (float(
        #                     corner[0]), float(
        #                     corner[1])), robotHandler.robot_pose)
        #             corners_relative.append(rel)
        #
        #         # Visualize corners on top camera
        #         frame_info.frame_top = visualize_xy_locations(
        #             frame_info.frame_top,
        #             corners_relative,
        #             robotHandler.robot_pose,
        #             is_top=True,
        #             color=(255, 0, 255),  # Magenta for zone corners
        #             radius=5,
        #             labels=[f"Z{zone_idx}.{i}" for i in range(len(corners))]
        #         )
        #
        #         # Visualize center on top camera
        #         frame_info.frame_top = visualize_xy_locations(
        #             frame_info.frame_top,
        #             [center_relative],
        #             robotHandler.robot_pose,
        #             is_top=True,
        #             color=(255, 255, 0),  # Cyan for zone center
        #             radius=8,
        #             labels=[f"Z{zone_idx}"]
        #         )
        #
        #         # Visualize corners on bottom camera
        #         frame_info.frame_bottom = visualize_xy_locations(
        #             frame_info.frame_bottom,
        #             corners_relative,
        #             robotHandler.robot_pose,
        #             is_top=False,
        #             color=(255, 0, 255),  # Magenta for zone corners
        #             radius=5,
        #             labels=[f"Z{zone_idx}.{i}" for i in range(len(corners))]
        #         )
        #
        #         # Visualize center on bottom camera
        #         frame_info.frame_bottom = visualize_xy_locations(
        #             frame_info.frame_bottom,
        #             [center_relative],
        #             robotHandler.robot_pose,
        #             is_top=False,
        #             color=(255, 255, 0),  # Cyan for zone center
        #             radius=8,
        #             labels=[f"Z{zone_idx}"]
        #         )

        main_profiler.record("visualization")

        main_profiler.end_frame()

    # Set the frame callback to use our processor
    computer_receiver.set_frame_callback(process)

    # Start keyboard input thread
    threading.Thread(
        target=inputProcessor.handleKeyboardMovementsLoop,
        daemon=True).start()

    # ----------------- START SERVERS -----------------
    if not computer_receiver.start_servers():
        return

    while True:
        print("connecting")
        if computer_receiver.wait_for_connection():
            print("Starting receive loop...")
            try:
                computer_receiver.receive_loop(
                    show_video=not args.no_display,
                    window_name_top=window_name_top,
                    window_name_bottom=window_name_bottom
                )
            except KeyboardInterrupt:
                print("\nSaving profiler data before exit...")
                robotHandler.profiler.save_profile()
                raise

        wait_time = config.RECONNECT_DELAY
        print(f"Waiting for {wait_time} seconds before reconnecting")
        time.sleep(wait_time)


if __name__ == "__main__":
    main()
