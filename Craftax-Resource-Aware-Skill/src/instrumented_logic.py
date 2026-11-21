def audit_mobs(rng, state, params, static_params):

    def _move_zombie(rng_and_state, zombie_index):
        rng, state = rng_and_state
        zombies = state.zombies
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(_rng, DIRECTIONS[1:5])
        random_move_proposed_position = zombies.position[zombie_index] + random_move_direction
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(state.player_position - zombies.position[zombie_index])
        player_move_direction_index_p = (player_move_direction_abs == player_move_direction_abs.max()) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(_rng, jnp.arange(2), p=player_move_direction_index_p)
        player_move_direction = player_move_direction.at[player_move_direction_index].set(jnp.sign(state.player_position[player_move_direction_index] - zombies.position[zombie_index][player_move_direction_index]).astype(jnp.int32))
        player_move_proposed_position = zombies.position[zombie_index] + player_move_direction
        close_to_player = jnp.sum(jnp.abs(zombies.position[zombie_index] - state.player_position)) < 10
        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(close_to_player, jax.random.uniform(_rng) < 0.75)
        proposed_position = jax.lax.select(close_to_player, player_move_proposed_position, random_move_proposed_position)
        is_attacking_player = jnp.sum(jnp.abs(zombies.position[zombie_index] - state.player_position)) == 1
        is_attacking_player = jnp.logical_and(is_attacking_player, zombies.attack_cooldown[zombie_index] <= 0)
        is_attacking_player = jnp.logical_and(is_attacking_player, zombies.mask[zombie_index])
        proposed_position = jax.lax.select(is_attacking_player, zombies.position[zombie_index], proposed_position)
        zombie_damage = jax.lax.select(state.is_sleeping, 7, 2)
        new_cooldown = jax.lax.select(is_attacking_player, 5, zombies.attack_cooldown[zombie_index] - 1)
        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)
        state = state.replace(player_health=state.player_health - zombie_damage * is_attacking_player, is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(is_attacking_player)), achievements=state.achievements.at[Achievement.WAKE_UP.value].set(jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_player)))
        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, proposed_position)
        position = jax.lax.select(valid_move, proposed_position, zombies.position[zombie_index])
        should_not_despawn = jnp.abs(zombies.position[zombie_index] - state.player_position).sum() < params.mob_despawn_distance
        rng, _rng = jax.random.split(rng)
        new_mob_map = state.mob_map.at[state.zombies.position[zombie_index, 0], state.zombies.position[zombie_index, 1]].set(jnp.logical_and(state.mob_map[state.zombies.position[zombie_index, 0], state.zombies.position[zombie_index, 1]], jnp.logical_not(zombies.mask[zombie_index])))
        new_mask = jnp.logical_and(state.zombies.mask[zombie_index], should_not_despawn)
        new_mob_map = new_mob_map.at[position[0], position[1]].set(jnp.logical_or(new_mob_map[position[0], position[1]], new_mask))
        state = state.replace(zombies=state.zombies.replace(position=state.zombies.position.at[zombie_index].set(position), attack_cooldown=state.zombies.attack_cooldown.at[zombie_index].set(new_cooldown), mask=state.zombies.mask.at[zombie_index].set(new_mask)), mob_map=new_mob_map)
        return ((_rng, state), None)
    rng, _rng = jax.random.split(rng)
    hp_before_zombie = state.player_health
    (rng, new_state), _ = jax.lax.scan(_move_zombie, (rng, state), jnp.arange(static_params.max_zombies))
    zombie_damage = hp_before_zombie - new_state.player_health

    def _move_cow(rng_and_state, cow_index):
        rng, state = rng_and_state
        cows = state.cows
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(_rng, DIRECTIONS[1:9])
        proposed_position = cows.position[cow_index] + random_move_direction
        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, proposed_position)
        position = jax.lax.select(valid_move, proposed_position, cows.position[cow_index])
        should_not_despawn = jnp.abs(cows.position[cow_index] - state.player_position).sum() < params.mob_despawn_distance
        new_mob_map = state.mob_map.at[state.cows.position[cow_index, 0], state.cows.position[cow_index, 1]].set(jnp.logical_and(state.mob_map[state.cows.position[cow_index, 0], state.cows.position[cow_index, 1]], jnp.logical_not(cows.mask[cow_index])))
        new_mask = jnp.logical_and(state.cows.mask[cow_index], should_not_despawn)
        new_mob_map = new_mob_map.at[position[0], position[1]].set(jnp.logical_or(new_mob_map[position[0], position[1]], new_mask))
        state = state.replace(cows=state.cows.replace(position=state.cows.position.at[cow_index].set(position), mask=state.cows.mask.at[cow_index].set(jnp.logical_and(state.cows.mask[cow_index], should_not_despawn))), mob_map=new_mob_map)
        return ((rng, state), None)
    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(_move_cow, (rng, new_state), jnp.arange(static_params.max_cows))

    def _move_skeleton(rng_and_state, skeleton_index):
        rng, state = rng_and_state
        skeletons = state.skeletons
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(_rng, DIRECTIONS[1:5])
        random_move_proposed_position = skeletons.position[skeleton_index] + random_move_direction
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(state.player_position - skeletons.position[skeleton_index])
        player_move_direction_index_p = (player_move_direction_abs == player_move_direction_abs.max()) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(_rng, jnp.arange(2), p=player_move_direction_index_p)
        player_move_direction = player_move_direction.at[player_move_direction_index].set(jnp.sign(state.player_position[player_move_direction_index] - skeletons.position[skeleton_index][player_move_direction_index]).astype(jnp.int32))
        player_move_towards_proposed_position = skeletons.position[skeleton_index] + player_move_direction
        player_move_away_proposed_position = skeletons.position[skeleton_index] - player_move_direction
        distance_to_player = jnp.sum(jnp.abs(skeletons.position[skeleton_index] - state.player_position))
        far_from_player = distance_to_player >= 10
        too_close_to_player = distance_to_player <= 3
        proposed_position = jax.lax.select(far_from_player, player_move_towards_proposed_position, random_move_proposed_position)
        proposed_position = jax.lax.select(too_close_to_player, player_move_away_proposed_position, proposed_position)
        rng, _rng = jax.random.split(rng)
        proposed_position = jax.lax.select(jax.random.uniform(_rng) > 0.85, proposed_position, random_move_proposed_position)
        is_attacking_player = jnp.logical_and(distance_to_player >= 4, distance_to_player <= 5)
        is_attacking_player = jnp.logical_or(is_attacking_player, jnp.logical_and(too_close_to_player, jnp.logical_not(is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, proposed_position))))
        is_attacking_player = jnp.logical_and(is_attacking_player, skeletons.attack_cooldown[skeleton_index] <= 0)
        is_attacking_player = jnp.logical_and(is_attacking_player, skeletons.mask[skeleton_index])
        can_spawn_arrow = state.arrows.mask.sum() < static_params.max_arrows
        new_arrow_position = skeletons.position[skeleton_index]
        is_spawning_arrow = jnp.logical_and(is_attacking_player, can_spawn_arrow)
        new_arrow_index = jnp.argmax(jnp.logical_not(state.arrows.mask))
        new_arrow_position = jax.lax.select(is_spawning_arrow, new_arrow_position, state.arrows.position[new_arrow_index])
        new_arrow_mask = jax.lax.select(is_spawning_arrow, True, state.arrows.mask[new_arrow_index])
        new_arrow_direction = jax.lax.select(is_spawning_arrow, player_move_direction, state.arrow_directions[new_arrow_index])
        new_arrows = state.arrows.replace(position=state.arrows.position.at[new_arrow_index].set(new_arrow_position), mask=state.arrows.mask.at[new_arrow_index].set(new_arrow_mask))
        state = state.replace(arrows=new_arrows, arrow_directions=state.arrow_directions.at[new_arrow_index].set(new_arrow_direction))
        proposed_position = jax.lax.select(is_attacking_player, skeletons.position[skeleton_index], proposed_position)
        new_cooldown = jax.lax.select(is_attacking_player, 4, skeletons.attack_cooldown[skeleton_index] - 1)
        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, proposed_position)
        position = jax.lax.select(valid_move, proposed_position, skeletons.position[skeleton_index])
        should_not_despawn = jnp.abs(skeletons.position[skeleton_index] - state.player_position).sum() < params.mob_despawn_distance
        new_mob_map = state.mob_map.at[state.skeletons.position[skeleton_index, 0], state.skeletons.position[skeleton_index, 1]].set(jnp.logical_and(state.mob_map[state.skeletons.position[skeleton_index, 0], state.skeletons.position[skeleton_index, 1]], jnp.logical_not(skeletons.mask[skeleton_index])))
        new_mask = jnp.logical_and(state.skeletons.mask[skeleton_index], should_not_despawn)
        new_mob_map = new_mob_map.at[position[0], position[1]].set(jnp.logical_or(new_mob_map[position[0], position[1]], new_mask))
        state = state.replace(skeletons=state.skeletons.replace(position=state.skeletons.position.at[skeleton_index].set(position), attack_cooldown=state.skeletons.attack_cooldown.at[skeleton_index].set(new_cooldown), mask=state.skeletons.mask.at[skeleton_index].set(jnp.logical_and(state.skeletons.mask[skeleton_index], should_not_despawn))), mob_map=new_mob_map)
        return ((rng, state), None)
    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(_move_skeleton, (rng, new_state), jnp.arange(static_params.max_skeletons))

    def _move_arrow(rng_and_state, arrow_index):
        rng, state = rng_and_state
        arrows = state.arrows
        proposed_position = arrows.position[arrow_index] + state.arrow_directions[arrow_index]
        proposed_position_in_player = (proposed_position == state.player_position).all()
        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_wall(state, proposed_position)
        in_wall = jnp.logical_and(in_wall, jnp.logical_not(state.map[proposed_position[0], proposed_position[1]] == BlockType.WATER.value))
        in_mob = is_in_mob(state, proposed_position)
        continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))
        position = proposed_position
        new_mask = jnp.logical_and(continue_move, arrows.mask[arrow_index])
        hit_player = jnp.logical_and(proposed_position_in_player, arrows.mask[arrow_index])
        hit_bench_or_furnace = jnp.logical_or(state.map[position[0], position[1]] == BlockType.FURNACE.value, state.map[position[0], position[1]] == BlockType.CRAFTING_TABLE.value)
        removing_block = jnp.logical_and(hit_bench_or_furnace, arrows.mask[arrow_index])
        new_block = jax.lax.select(removing_block, BlockType.PATH.value, state.map[position[0], position[1]])
        state = state.replace(arrows=state.arrows.replace(position=state.arrows.position.at[arrow_index].set(position), mask=state.arrows.mask.at[arrow_index].set(new_mask)), player_health=state.player_health - 2 * hit_player, is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)), map=state.map.at[position[0], position[1]].set(new_block))
        return ((rng, state), None)
    rng, _rng = jax.random.split(rng)
    hp_before_arrow = new_state.player_health
    (rng, new_state), _ = jax.lax.scan(_move_arrow, (rng, new_state), jnp.arange(static_params.max_arrows))
    arrow_damage = hp_before_arrow - new_state.player_health
    return (new_state, zombie_damage, arrow_damage)

def audit_craftax_step(rng, state, action, params, static_params):
    init_achievements = state.achievements
    init_health = state.player_health
    action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
    state = do_crafting(state, action)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, action, static_params)
    state = place_block(state, action, static_params)
    state = move_player(state, action)
    rng, _rng = jax.random.split(rng)
    state, zombie_damage, arrow_damage = audit_mobs(_rng, state, params, static_params)
    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)
    state = update_plants(state, static_params)
    hp_before_intrinsic = state.player_health
    state = update_player_intrinsics(state, action)
    intrinsic_health_delta = state.player_health - hp_before_intrinsic
    state = cap_inventory(state)
    hp_before_cap = state.player_health
    lava_at_cap = state.map[state.player_position[0], state.player_position[1]] == 14
    state = update_health(state)
    cap_health_delta = state.player_health - hp_before_cap
    achievement_reward = state.achievements.astype(jnp.float32).sum() - init_achievements.astype(jnp.float32).sum()
    health_reward = (state.player_health - init_health) * 0.1
    reward = achievement_reward + health_reward
    rng, _rng = jax.random.split(rng)
    state = state.replace(timestep=state.timestep + 1, light_level=calculate_light_level(state.timestep + 1, params), state_rng=_rng)
    return (state, reward, jnp.array([zombie_damage, arrow_damage, intrinsic_health_delta, cap_health_delta, hp_before_cap, lava_at_cap], jnp.float32))