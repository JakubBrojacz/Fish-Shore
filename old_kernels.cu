//__global__ void
//prepare_k(cData* f, int dx, int dy,
//	float dt, int lb)
//{
//	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
//	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
//
//	if (gtidx < dx && gtidy < dy)
//	{
//		int fj = gtidx + gtidy * dx;
//		f[fj].x = 0;
//		f[fj].y = 0;
//		f[fj].z = 0;
//	}
//}
//
//
//__global__ void
//cohesion_k(cData* part, cData* v, cData* f, int dx, int dy,
//	float dt, int lb)
//{
//	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
//	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
//
//	cData pterm;
//
//	if (gtidx < dx && gtidy < dy)
//	{
//		int fj = gtidx + gtidy * dx;
//		pterm = part[6 * fj];
//
//		int count = 0;
//		cData mid = empty();
//		for (int i = 0; i < SHORE; i++)
//			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
//			{
//				count++;
//				mid = add(mid, part[6 * i]);
//			}
//		if (count > 0)
//		{
//			mid = divide(mid, count);
//			cData des = subtract(mid, pterm);
//
//			cData steer = limit(des, MAX_FORCE);
//			steer = multiply(steer, COH_MULTI);
//
//			f[fj] = add(f[fj], steer);
//		}
//	}
//}
//
//__global__ void
//separation_k(cData* part, cData* v, cData* f, int dx, int dy,
//	float dt, int lb)
//{
//	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
//	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
//
//	cData pterm;
//
//	if (gtidx < dx && gtidy < dy)
//	{
//		int fj = gtidx + gtidy * dx;
//		pterm = part[6 * fj];
//
//		int count = 0;
//		cData mid = empty();
//		for (int i = 0; i < SHORE; i++)
//			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS)
//			{
//				count++;
//
//				//tmp = SEPARATION_RADIUS - abs(pterm - part[6 * i])
//				cData tmp = subtract(pterm, part[6 * i]);
//				tmp = abs(tmp);
//				tmp = multiply(tmp, -1);
//				tmp = add(tmp, SEPARATION_RADIUS);
//
//				if (getSquaredDistance(part[6 * i], pterm) < SEPARATION_RADIUS * SEPARATION_RADIUS / 100)
//				{
//					tmp = multiply(tmp, 3);
//				}
//				mid.x += tmp.x * (pterm.x > part[6 * i].x ? 1 : -1);
//				mid.y += tmp.y * (pterm.y > part[6 * i].y ? 1 : -1);
//				mid.z += tmp.z * (pterm.z > part[6 * i].z ? 1 : -1);
//			}
//		if (count > 0)
//		{
//			mid = divide(mid, count);
//
//			cData steer = setMagnitude(mid, MAX_FORCE);
//			steer = multiply(steer, SEP_MULTI);
//
//			f[fj] = add(f[fj], steer);
//		}
//	}
//}
//
//
//__global__ void
//alignment_k(cData* part, cData* v, cData* f, int dx, int dy,
//	float dt, int lb)
//{
//	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
//	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
//
//	cData pterm;
//
//	if (gtidx < dx && gtidy < dy)
//	{
//		int fj = gtidx + gtidy * dx;
//		pterm = part[6 * fj];
//
//		int count = 0;
//		cData mid = empty();
//		for (int i = 0; i < SHORE; i++)
//			if (i != fj && getSquaredDistance(part[6 * i], pterm) < SIGN_RADIUS * SIGN_RADIUS)
//			{
//				count++;
//				mid = add(mid, v[i]);
//			}
//		if (count > 0)
//		{
//			mid = divide(mid, count);
//
//			cData steer = setMagnitude(mid, MAX_FORCE);
//			steer = multiply(steer, ALI_MULTI);
//
//			f[fj] = add(f[fj], steer);
//		}
//	}
//}
//
