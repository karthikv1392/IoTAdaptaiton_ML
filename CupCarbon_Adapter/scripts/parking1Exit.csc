set mod 0
set counter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	plus counter $sensorVal1 $counter
	data p s34 $sensorVal1
	function y adapter ada,100,10
	if($y==10.0)
		send $p 11
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 47
	end
	while($sensorVal1<5.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s33 $sensorVal1
		function y adapter ada,100,10
		if($y==10.0)
			if($counter>=0)
				send A 35
			else
				send N 35
			end
		send $p 11
		end
		if($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 47
		end
		delay 30000
	end
	if($sensorVal1>=5.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=5.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s33 $sensorVal1
		function y adapter ada,100,10
		if($y==10.0)
			if($counter>=0)
				send A 35
			else
				send N 35
			end
		send $p 11
		end
		if($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 47
		end
		send $p 47
		delay 10000
	end
	if($sensorVal1<5.0)
		set mod 0
	end
end
