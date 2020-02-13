set mod 0
set counter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	plus counter $sensorVal1 $counter
	data p s42 $sensorVal1
	function y adapter ada,100,10
	if($y==10.0)
		send $p 11
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 48
	end
	while($sensorVal1<12.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter
		data p s42 $sensorVal1
		function y adapter ada,100,10
		if ($y==10.0)
			if($counter>=300.0)
				send N 43
			else
				send A 43
			end
		send $p 11
		end
		if ($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 48
		end
		delay 20000
	end
	if($sensorVal1>=12.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=12.0)
		areadsensor var
		rdata $var t x sensorVal1
		plus counter $sensorVal1 $counter	
		data p s42 $sensorVal1
		function y adapter ada,100,10
		if ($y==10.0)
			if($counter>=300.0)
				send N 43
			else
				send A 43
			end
		send $p 11
		end
		if ($y==20.0)
			send $p 11
		end
		if ($y==30.0)
			send $p 48
		end
		delay 5000
	end
	if($sensorVal1<12.0)
		set mod 0
	end
end
