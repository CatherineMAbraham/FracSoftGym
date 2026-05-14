from gymnasium.envs.registration import register

# Version 0 (The old commit)
register(
    id="anklesurg-v0",
    entry_point="gym_fracture.versions.v0.fracuresurgery_v0:fracturesurgery_env_v0",
)

# Version 1 (The intermediate version)
register(
    id="anklesurg-v1",
    entry_point="gym_fracture.versions.v1.fracuresurgery_v1:fracturesurgery_env_v1",
)

# Version 2 (Your newest development)
register(
    id="anklesurg-v2",
    entry_point="gym_fracture.versions.v2.fracuresurgery_v2:fracturesurgery_env_v2",
)
