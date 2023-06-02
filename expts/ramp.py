import argparse
def create_ramp(start, end, num_steps):
    ramp = [start]
    value = start

    for i in range(1, num_steps+1):
        value = value / 2
        ramp.append(value)

    ramp.append(end)
    
    return ramp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=float, default=100, help='start value')
    parser.add_argument('--end', type=float, default=0, help='end value')
    parser.add_argument('--steps', type=int, default=1, help='number of steps')
    args = parser.parse_args()
    
    ramp = create_ramp(args.start, args.end, args.steps)
    print(ramp)
