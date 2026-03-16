import torch

from src.solver import runge_kutta

class PendulumOnRail:
    def __init__(self, mass_pendulum: float, mass_wagon: float, radius: float, batch_size: int = 1, randomized_state: bool = False, device: str = "cpu"):
        self._device = device
        self._batch_size = batch_size
        self._mass_pendulum = mass_pendulum
        self._mass_wagon = mass_wagon
        self._radius = radius
        self._gravity = 9.81

        self._surface_wagon = 0.002
        self._surface_pendulum = 0.002

        self._xmin = -2
        self._xmax = 2
        self._bounce_coefficient = 0.7

        self._push_force = torch.zeros((batch_size))
        self._state = torch.zeros((batch_size, 4))
        """
        (batch_size, [x(t), x'(t), theta(t) (0 = equilibrium), theta'(t)])
        """

        if randomized_state:
            self._state = torch.rand((batch_size, 4)) \
                * torch.tensor((0, 0, torch.pi, torch.pi / 2)) \
                - torch.tensor((0, 0, torch.pi / 2, torch.pi / 4))
        
        self._state = self._state.to(device=self._device)
        self._clamp_state_min = torch.tensor((-2, -10, -torch.inf, -12 * torch.pi), device=self._device)
        self._clamp_state_max = torch.tensor((2, 10, torch.inf, 12 * torch.pi), device=self._device)
        
    def get_batch_size(self) -> float:
        return self._batch_size
    
    def get_mass_pendulum(self) -> float:
        return self._mass_pendulum
    
    def set_mass_pendulum(self, mass_pendulum: float):
        self._mass_pendulum = mass_pendulum
    
    def get_mass_wagon(self) -> float:
        return self._mass_wagon
    
    def set_mass_wagon(self, mass_wagon: float):
        self._mass_wagon = mass_wagon
    
    def get_gravity(self) -> float:
        return self._gravity
    
    def set_gravity(self, gravity: float):
        self._gravity = gravity
    
    def get_radius(self) -> float:
        return self._radius
    
    def set_radius(self, radius: float):
        self._radius = radius
    
    def get_surface_wagon(self) -> float:
        return self._surface_wagon
    
    def set_surface_wagon(self, surface_wagon: float):
        self._surface_wagon = surface_wagon
    
    def get_surface_pendulum(self) -> float:
        return self._surface_pendulum
    
    def set_surface_pendulum(self, surface_pendulum: float):
        self._surface_pendulum = surface_pendulum
    
    def get_xlim(self) -> torch.Tensor:
        return torch.tensor((self._xmin, self._xmax))
    
    def set_xlim(self, xmin: float = -float("inf"), xmax: float = float("inf")):
        self._xmin = xmin
        self._xmax = xmax
    
    def get_bounce_coefficient(self) -> float:
        return self._bounce_coefficient
    
    def set_bounce_coefficient(self, bounce_coefficient: float):
        self._bounce_coefficient = bounce_coefficient
    
    def get_push_force(self) -> torch.Tensor:
        return self._push_force
    
    def set_push_force(self, push_force: torch.Tensor):
        self._push_force = push_force
    
    def get_state(self) -> torch.Tensor:
        return self._state
    
    def set_state(self, state: torch.Tensor):
        self._state = state
    
    def derivate(self, t: float, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            state = self._state
        
        with torch.no_grad():
            x_pos = state[:, 0]
            x_spd = state[:, 1]
            theta_pos = state[:, 2]
            theta_spd = state[:, 3]

            m0 = self._mass_wagon
            m1 = self._mass_pendulum
            p = self._push_force
            g = self._gravity
            r = self._radius

            sin_theta = torch.sin(theta_pos)
            cos_theta = torch.cos(theta_pos)
            m1_sin_theta = m1 * sin_theta
            
            x_acc = (p + m1_sin_theta * (g * cos_theta - r * theta_spd * theta_spd)) / (m0 + m1_sin_theta)
            theta_acc = -(g * sin_theta + x_acc * cos_theta) / r
            return torch.stack((x_spd, x_acc, theta_spd, theta_acc), dim=1)

    def next_state(self, dt: float, push_force: torch.Tensor):
        self.set_push_force(push_force)
        self._state = runge_kutta(0, self._state, dt, self.derivate)

        self._state = torch.clamp(self._state, self._clamp_state_min, self._clamp_state_max)

        xmin_condition = (self._state[:, 0] < self._xmin)
        self._state[:, 0] = torch.where(xmin_condition, self._xmin, self._state[:, 0])
        self._state[:, 1] = torch.where(xmin_condition, -self._bounce_coefficient * self._state[:, 1], self._state[:, 1])
        xmax_condition = (self._state[:, 0] > self._xmax)
        self._state[:, 0] = torch.where(xmax_condition, self._xmax, self._state[:, 0])
        self._state[:, 1] = torch.where(xmax_condition, -self._bounce_coefficient * self._state[:, 1], self._state[:, 1])


class PendulumOnRailWithFriction:
    def __init__(self, mass_pendulum: float, mass_wagon: float, radius: float, batch_size: int = 1, randomized_state: bool = False, device: str = "cpu"):
        self._device = device
        self._batch_size = batch_size
        self._mass_pendulum = mass_pendulum
        self._mass_wagon = mass_wagon
        self._radius = radius
        self._gravity = 9.81

        self._air_friction = 0
        self._surface_wagon = 0.01
        self._surface_pendulum = 0.002
        
        self._xmin = -2
        self._xmax = 2
        self._bounce_coefficient = 0.7

        self._push_force = torch.zeros((batch_size))
        self._state = torch.zeros((batch_size, 4))
        """
        (batch_size, [x(t), x'(t), theta(t) (0 = equilibrium), theta'(t)])
        """

        if randomized_state:
            self._state = torch.rand((batch_size, 4)) \
                * torch.tensor((0, 0, torch.pi, torch.pi / 2)) \
                - torch.tensor((0, 0, torch.pi / 2, torch.pi / 4))
        
        self._state = self._state.to(device=self._device)
        self._clamp_state_min = torch.tensor((-torch.inf, -100, -torch.inf, -12 * torch.pi), device=self._device)
        self._clamp_state_max = torch.tensor((torch.inf, 100, torch.inf, 12 * torch.pi), device=self._device)
        
    def get_batch_size(self) -> float:
        return self._batch_size
    
    def get_mass_pendulum(self) -> float:
        return self._mass_pendulum
    
    def set_mass_pendulum(self, mass_pendulum: float):
        self._mass_pendulum = mass_pendulum
    
    def get_mass_wagon(self) -> float:
        return self._mass_wagon
    
    def set_mass_wagon(self, mass_wagon: float):
        self._mass_wagon = mass_wagon
    
    def get_gravity(self) -> float:
        return self._gravity
    
    def set_gravity(self, gravity: float):
        self._gravity = gravity
    
    def get_radius(self) -> float:
        return self._radius
    
    def set_radius(self, radius: float):
        self._radius = radius
    
    def get_air_friction(self) -> float:
        return self._air_friction
    
    def set_air_friction(self, air_friction: float):
        self._air_friction = air_friction
    
    def get_surface_wagon(self) -> float:
        return self._surface_wagon
    
    def set_surface_wagon(self, surface_wagon: float):
        self._surface_wagon = surface_wagon
    
    def get_surface_pendulum(self) -> float:
        return self._surface_pendulum
    
    def set_surface_pendulum(self, surface_pendulum: float):
        self._surface_pendulum = surface_pendulum
    
    def get_xlim(self) -> torch.Tensor:
        return torch.tensor((self._xmin, self._xmax))
    
    def set_xlim(self, xmin: float = -float("inf"), xmax: float = float("inf")):
        self._xmin = xmin
        self._xmax = xmax
    
    def get_bounce_coefficient(self) -> float:
        return self._bounce_coefficient
    
    def set_bounce_coefficient(self, bounce_coefficient: float):
        self._bounce_coefficient = bounce_coefficient

    def get_push_force(self) -> torch.Tensor:
        return self._push_force
    
    def set_push_force(self, push_force: torch.Tensor):
        self._push_force = torch.flatten(push_force)
    
    def get_state(self) -> torch.Tensor:
        return self._state
    
    def set_state(self, state: torch.Tensor):
        self._state = state
    
    def derivate(self, t: float, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            state = self._state
        
        with torch.no_grad():
            x_pos = state[:, 0]
            x_spd = state[:, 1]
            theta_pos = state[:, 2]
            theta_spd = state[:, 3]

            m0 = self._mass_wagon
            m1 = self._mass_pendulum
            p = self._push_force
            g = self._gravity
            r = self._radius
            f = self._air_friction
            s0 = self._surface_wagon
            s1 = self._surface_pendulum

            sin_theta = torch.sin(theta_pos)
            cos_theta = torch.cos(theta_pos)
            m1_r_theta_spd_squared = (m1 * r) * theta_spd * theta_spd

            num = sin_theta * (m1_r_theta_spd_squared - (f * s1) * x_spd * sin_theta) - (f * s0) * x_spd + p
            den = m0 + m1 * sin_theta * sin_theta
            x_acc = num / den
            
            num = cos_theta * ((f * (s0 - m0 / m1 * s1)) * x_spd - m1_r_theta_spd_squared * sin_theta - p) - ((m0 + m1) * g) * sin_theta
            theta_acc = num / (r * den) - (f * s1 / m1) * theta_spd
            return torch.stack((x_spd, x_acc, theta_spd, theta_acc), dim=1)

    def next_state(self, dt: float, push_force: torch.Tensor) -> torch.Tensor:
        """
        returns a tensor of 0 if there is a bounce, 1 else
        """
        self.set_push_force(push_force)
        self._state = runge_kutta(0, self._state, dt, self.derivate)
        self._state = torch.clamp(self._state, self._clamp_state_min, self._clamp_state_max)
        
        two_pi = 2 * torch.pi
        theta_pos = self._state[:, 2]
        theta_pos = torch.where(theta_pos < two_pi, theta_pos + two_pi, theta_pos)
        theta_pos = torch.where(theta_pos > two_pi, theta_pos - two_pi, theta_pos)

        xmin_condition = (self._state[:, 0] < self._xmin)
        self._state[:, 0] = torch.where(xmin_condition, self._xmin, self._state[:, 0])
        self._state[:, 1] = torch.where(xmin_condition, -self._bounce_coefficient * self._state[:, 1], self._state[:, 1])
        xmax_condition = (self._state[:, 0] > self._xmax)
        self._state[:, 0] = torch.where(xmax_condition, self._xmax, self._state[:, 0])
        self._state[:, 1] = torch.where(xmax_condition, -self._bounce_coefficient * self._state[:, 1], self._state[:, 1])
        return torch.where(xmax_condition | xmin_condition, 1, 0)

