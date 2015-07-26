#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DDP_TASK_ID,
  REDUCE_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
  int num_elements = 12; 
  int num_subregions = 4;
  int nr, sr; // number of regions, size of regions
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-r"))
        num_subregions = atoi(command_args.argv[++i]);
    }
  }
  if ((num_elements % num_subregions) != 0) {
    printf("Number of regions must divide number of elements. Error,exit.\n");
    exit(1);
  }
  printf("Running ddp for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);
  sr = num_elements / num_subregions;
  nr = num_subregions-1;
  printf("Expected outcome (nr: %d, sr: %d): %e\n", 
	 nr, sr, sr * nr * (1.0/6.0 + nr * (1.0/2.0 + nr/3.0)));

  // Create a logical region for the input.
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace input_is = runtime->create_index_space(ctx, 
                          Domain::from_rect<1>(elem_rect));
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, input_is, input_fs);

  // Create a logical region for partial sums.
  Rect<1> elem_rect_output(Point<1>(0),Point<1>(num_subregions-1));
  IndexSpace output_is = runtime->create_index_space(ctx, 
                          Domain::from_rect<1>(elem_rect_output));
  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
  }
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, output_fs);

  // Arrange parallel processing of each subregion
  Rect<1> color_bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  IndexPartition input_ip, output_ip;
  Blockify<1> coloring(num_elements/num_subregions);
  input_ip = runtime->create_index_partition(ctx, input_is, coloring);
  Blockify<1> coloring2(1);
  output_ip = runtime->create_index_partition(ctx, output_is, coloring2);

  // change to partitions
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, input_ip);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);

  Domain launch_domain = color_domain; 
  ArgumentMap arg_map;

  // Prepare for the initialization actor
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, launch_domain, 
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/, 
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(0, FID_X);
  runtime->execute_index_space(ctx, init_launcher);

  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_launcher);

  // Prepare for the dot product compute actor
  IndexLauncher ddp_launcher(DDP_TASK_ID, launch_domain, 
			     TaskArgument(NULL, 0), arg_map);
  ddp_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  ddp_launcher.region_requirements[0].add_field(FID_X);
  ddp_launcher.region_requirements[0].add_field(FID_Y);
  ddp_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  ddp_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_index_space(ctx, ddp_launcher);
                    
  TaskLauncher reduce_launcher(REDUCE_TASK_ID, TaskArgument(NULL, 0));
  reduce_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  reduce_launcher.region_requirements[0].add_field(FID_X);
  reduce_launcher.region_requirements[0].add_field(FID_Y);
  reduce_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  reduce_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_task(ctx, reduce_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, input_is);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    acc.write(DomainPoint::from_point<1>(pir.p), (double)point);
  }
}

void ddp_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime) {
  double partial_ddp = 0;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == 0);
  const int point = task->index_point.point_data[0];

  RegionAccessor<AccessorType::Generic, double> acc_x = 
    regions[0].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_y = 
    regions[0].get_field_accessor(FID_Y).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_z = 
    regions[1].get_field_accessor(FID_Z).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    partial_ddp += acc_x.read(DomainPoint::from_point<1>(pir.p)) * 
                           acc_y.read(DomainPoint::from_point<1>(pir.p));
  }

  // fill the reduced region
  Domain dom2 = runtime->get_index_space_domain(ctx, 
      task->regions[1].region.get_index_space());
  Rect<1> rect2 = dom2.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect2); pir; pir++) {
    acc_z.write(DomainPoint::from_point<1>(pir.p), partial_ddp);
    printf("Stored partial ddp for subregion %d as %e\n", point, partial_ddp);
  }
}

void reduce_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == 0);
  RegionAccessor<AccessorType::Generic, double> acc_z = 
    regions[1].get_field_accessor(FID_Z).typeify<double>();
  printf("Reducing results...");
  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[1].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  double sum = 0;
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    sum += acc_z.read(DomainPoint::from_point<1>(pir.p));
  }
  printf("ddp equals: %e\n", sum);
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<ddp_task>(DDP_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<reduce_task>(REDUCE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);

  return HighLevelRuntime::start(argc, argv);
}
