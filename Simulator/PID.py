import math

def _clamp(value, limits):
	lower, upper = limits
	if value is None:
		return None
	elif upper is not None and value > upper:
		return upper
	elif lower is not None and value < lower:
		return lower
	return value

class PID(object):
	"""
	A simple PID controller. No fuss.
	"""

	def __init__(self,
				 Kf=0.0, Kp=1.0, Ki=0.0, Kd=0.0,
				 setpoint=0,
				 output_limits=(None, None)):
		"""
		:param Kf: The value for the feedforward gain Kf
		:param Kp: The value for the proportional gain Kp
		:param Ki: The value for the integral gain Ki
		:param Kd: The value for the derivative gain Kd
		:param setpoint: The initial setpoint that the PID will try to achieve
		:param output_limits: The initial output limits to use, given as an iterable with 2 elements, for example:
							  (lower, upper). The output will never go below the lower limit or above the upper limit.
							  Either of the limits can also be set to None to have no limit in that direction. Setting
							  output limits also avoids integral windup, since the integral term will never be allowed
							  to grow outside of the limits.
		"""
		self.Kf, self.Kp, self.Ki, self.Kd = Kf, Kp, Ki, Kd
		self.setpoint = setpoint

		self._min_output, self._max_output = output_limits
		self._last_time = None
		self._last_dt = None

		self.reset()

	def update(self, input_, time):
		"""
		update the PID controller with *input_* and calculate and return a control output if sample_time seconds has
		passed since the last update. If no new output is calculated, return the previous output instead (or None if
		no value has been calculated yet).
		:param dt: If set, uses this value for timestep instead of real time. This can be used in simulations when
				   simulation time is different from real time.
		"""
		# compute error terms
		error = self.setpoint - input_
		# errror = math.cos(error)
		d_input = input_ - (self._last_input if self._last_input is not None else input_)

		self._feedforward = self.Kf * input_

		#compute proportional term
		self._proportional = self.Kp * error

		if self._last_time != None and self._last_time < time:
			if self._last_dt != None and self._last_time == time:
				dt = self._last_dt
			else:
				dt = time - self._last_time
				self._last_dt = dt

			print(dt)
			# compute integral and derivative terms
			self._integral += self.Ki * error * dt
			self._integral = _clamp(self._integral, self.output_limits)  # avoid integral windup

			self._derivative = -self.Kd * d_input / dt
		else:
			self._integral = 0
			self._derivative = 0

		# compute final output
		output = self._feedforward + self._proportional + self._integral + self._derivative
		output = _clamp(output, self.output_limits)

		# keep track of state
		self._last_output = output
		self._last_input = input_
		self._last_time = time
		return output

	def setSetpoint(self, setpoint):
		"""
		Give the pid controller a new setpoint
		"""
		self.setpoint = setpoint
	
	def components(self):
		"""
		The P-, I- and D-terms from the last computation as separate components as a tuple. Useful for visualizing
		what the controller is doing or when tuning hard-to-tune systems.
		"""
		return self._feedforward, self._proportional, self._integral, self._derivative

	@property
	def tunings(self):
		"""The tunings used by the controller as a tuple: (Kp, Ki, Kd)"""
		return self._feedforward, self.Kp, self.Ki, self.Kd

	@tunings.setter
	def tunings(self, tunings):
		"""Setter for the PID tunings"""
		self._feedforward, self.Kp, self.Ki, self.Kd = tunings

	@property
	def output_limits(self):
		"""
		The current output limits as a 2-tuple: (lower, upper). See also the *output_limts* parameter in
		:meth:`PID.__init__`.
		"""
		return self._min_output, self._max_output

	@output_limits.setter
	def output_limits(self, limits):
		"""Setter for the output limits"""
		if limits is None:
			self._min_output, self._max_output = None, None
			return

		min_output, max_output = limits

		if None not in limits and max_output < min_output:
			raise ValueError('lower limit must be less than upper limit')

		self._min_output = min_output
		self._max_output = max_output

		self._integral = _clamp(self._integral, self.output_limits)
		self._last_output = _clamp(self._last_output, self.output_limits)

	def reset(self):
		"""
		Reset the PID controller internals, setting each term to 0 as well as cleaning the integral,
		the last output and the last input (derivative calculation).
		"""
		self._proportional = 0
		self._integral = 0
		self._derivative = 0

		self._last_output = None
		self._last_input = None