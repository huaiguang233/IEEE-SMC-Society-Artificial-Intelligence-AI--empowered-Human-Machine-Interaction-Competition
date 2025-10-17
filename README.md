# IEEE-SMC-Society-Artificial-Intelligence-AI--empowered-Human-Machine-Interaction-Competition
Team NPU

Our main idea was to enable three robots to collaboratively build an occupancy map, where each robot prioritizes exploring areas that have not yet been covered by others, while simultaneously detecting victims. Upon detecting a victim, the robot would broadcast the victim’s coordinates, and the team would then perform a bidding process to determine which robot should approach the victim, based on the path cost computed from global occupancy map.

However, we heard about the competition only a few days before the submission deadline, which made it impossible to implement the full concept.
In the file Multi_robot_collaborative_mapping.py, we implemented the collaborative mapping component, which allows multiple robots to share map information.
Due to time constraints, we realized we could not complete the entire multi-robot coordination framework before the deadline. Therefore, in proposed_solution.py, we implemented a simplified version: each robot independently constructs its own occupancy map, detects and records victim locations, and stops its movement to notify the human supervisor once it reaches the vicinity of a detected victim.

We have made our best effort within the available time.
