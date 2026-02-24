#ifndef RUNTIME_C
#define RUNTIME_C


#include <float.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/// COMMON


#define UNUSED(x) (void)(x)


typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef size_t usize;

typedef float f32;
typedef double f64;


#define ANSI_RED   "\033[91m"
#define ANSI_BLUE  "\033[94m"
#define ANSI_PINK  "\033[95m"
#define ANSI_RESET "\033[0m"

#define C(c, s) c s ANSI_RESET

#define C_RED(s)  C (ANSI_RED,  s)
#define C_BLUE(s) C (ANSI_BLUE, s)
#define C_PINK(s) C (ANSI_PINK, s)


void *
malloc0 (usize size);


#define LOCATION_NONE "???"


extern const char *location;


#define CSTACK_SIZE (1024 * 0xFF)
#define VSTACK_SIZE (1024 * 0xFF)
#define SSTACK_SIZE (1024 * 0xFF)


void
cstack_push (const char *c);

void
cstack_pop (void);

usize
cstack_size (void);


struct value_trace;
struct value;


static struct value *vstack_last_pop = NULL;


void
vstack_push (struct value *v);

struct value *
vstack_pop (void);

usize
vstack_size (void);

#ifdef DEBUG
void
vstack_append_trace (struct value_trace *trace);
#endif


void
sstack_push (struct value *v);

void
sstack_pop (void);

usize
sstack_size (void);


/// PANIC


void
panic_va (const char *format, va_list va);

void
panic (const char *format, ...);


#define panic_assert(condition)                                                \
  if (condition)                                                               \
    ;                                                                          \
  else                                                                         \
    panic ("assertion `%s' has failed", #condition)


void
warning_va (const char *format, va_list va);

void
warning (const char *format, ...);


/// GARBAGE COLLECTOR


void
gc_register (struct value *value);

void
gc_destroy (void);

void
gc_mark (void);

void
gc_sweep (void);

void
gc_collect (void);

void
gc_collect_try (void);


/// CLOSURE


typedef void (closure_f) (struct value **);


struct closure
{
  struct value **env;

  usize env_size;

  closure_f *f;
};


struct closure *
closure_create (struct value **env, usize env_size, closure_f *f);

void
closure_destroy (struct closure *closure);

void
closure_mark (struct closure *closure);


/// VALUE


typedef void (block_f) (void);


enum value_type
{
  VALUE_U,
  VALUE_F,
  VALUE_B,
  VALUE_C,
};


#ifdef DEBUG
struct value_trace
{
  struct value_trace *next;

  const char *location;
  const char *transform;
};


struct value_trace *
value_trace_create (const char *location, const char *transform)
{
  struct value_trace *trace = malloc0 (sizeof (struct value_trace));

  trace->location = location;
  trace->transform = transform;

  return trace;
}


void
value_trace_destroy (struct value_trace *trace)
{
  free (trace);
}


void
value_trace_append (struct value_trace **head, struct value_trace *node)
{
  if (*head == NULL)
    {
      *head = node;
      return;
    }

  struct value_trace *current = *head;

  while (current->next != NULL)
    {
      current = current->next;
    }

  current->next = node;
}
#endif


struct value
{
  struct value *next;

  union
  {
    f64 f;

    block_f *b;

    struct closure *c;
  } d;

  enum value_type t;

  bool marked;

#ifdef DEBUG
  struct value_trace *debug_trace;
#endif
};


struct value *
value_create (enum value_type t);

// struct value *
// value_copy (struct value *value);

void
value_destroy (struct value *value);

void
value_mark (struct value *value);

void
value_unmark (struct value *value);

struct value *
value_box_u (void);

struct value *
value_box_f (f64 f);

struct value *
value_box_b (block_f *b);

struct value *
value_box_c (struct value *env[], usize env_size, closure_f *f);

void
value_unbox_u (struct value *value);

f64
value_unbox_f (struct value *value);

block_f *
value_unbox_b (struct value *value);

struct closure *
value_unbox_c (struct value *value);

void
value_fprint (FILE *stream, struct value *value);

void
value_fprintn (FILE *stream, struct value *value);

void
value_print (struct value *value);

void
value_printn (struct value *value);

bool
value_bool (struct value *value);

void
value_execute (struct value *value);


#ifdef DEBUG
void
value_append_trace (struct value *value, struct value_trace *trace);
#endif



/// COMMON


void *
malloc0 (usize size)
{
  return calloc (1, size);
}


const char *location = LOCATION_NONE;


static const char *cstack[CSTACK_SIZE];

static usize cstack_head;


void
cstack_push (const char *c)
{
#ifdef DEBUG
  panic_assert (cstack_head < CSTACK_SIZE);
#endif

  cstack[cstack_head++] = c;
}


void
cstack_pop (void)
{
#ifdef DEBUG
  panic_assert (cstack_head > 0);
#endif

  cstack_head--;
}


usize
cstack_size (void)
{
  return cstack_head;
}


static struct value *vstack[VSTACK_SIZE];

static usize vstack_head;


void
vstack_push (struct value *v)
{
#ifdef DEBUG
  panic_assert (vstack_head < VSTACK_SIZE);
#endif

  vstack[vstack_head++] = v;
}


struct value *
vstack_pop (void)
{
#ifdef DEBUG
  panic_assert (vstack_head > 0);
#endif

  return vstack_last_pop = vstack[--vstack_head];
}


usize
vstack_size (void)
{
  return vstack_head;
}


#ifdef DEBUG
void
vstack_append_trace (struct value_trace *trace)
{
  value_append_trace (vstack[vstack_head - 1], trace);
}
#endif


static struct value *sstack[SSTACK_SIZE];

static usize sstack_head;


void
sstack_push (struct value *v)
{
#ifdef DEBUG
  panic_assert (sstack_head < SSTACK_SIZE);
#endif

  sstack[sstack_head++] = v;
}

void
sstack_pop (void)
{
#ifdef DEBUG
  panic_assert (sstack_head > 0);
#endif

  sstack_head--;
}


usize
sstack_size (void)
{
  return sstack_head;
}


/// PANIC


void
panic_va (const char *format, va_list va)
{
  fprintf (stderr, C_RED ("PANIC") " ");

  vfprintf (stderr, format, va);

  fprintf (stderr, "\n\n  Call Stack (most recent call first):\n");

  if (cstack_head)
    for (usize i = cstack_head; i-- > 0;)
      {
        if (i == cstack_head - 1)
          {
            fprintf (stderr, "    in [%ld] " C_BLUE ("%s"), i, cstack[i]);

            fprintf (stderr, " at " C_BLUE ("%s") "\n", location);
          }
        else
          fprintf (stderr, "       [%ld] " C_BLUE ("%s") "\n", i, cstack[i]);
      }
  else
    {
      fprintf (stderr, "    " C_BLUE ("()"));

      fprintf (stderr, " at " C_BLUE ("%s") "\n", location);
    }

  fprintf (stderr, "\n");

  fprintf (stderr, "  Most recent pop:\n");

  if (vstack_last_pop)
    {
      fprintf (stderr, "    ");
      value_fprintn (stderr, vstack_last_pop);
    }

  fprintf (stderr, "\n");

  fprintf (stderr, "  Virtual Stack (most recent value first):\n");

  for (usize i = vstack_head; i-- > 0;)
    {
      fprintf (stderr, "       [%ld] ", i);
      value_fprintn (stderr, vstack[i]);
    }

  fprintf (stderr, "\n");

  exit (1);
  // abort ();
}


void
panic (const char *format, ...)
{
  va_list va;

  va_start (va, format);

  panic_va (format, va);

  va_end (va);
}


void
warning_va (const char *format, va_list va)
{
  fprintf (stderr, C_PINK ("WARNING") " ");

  vfprintf (stderr, format, va);

  fprintf (stderr, "\n");
}


void
warning (const char *format, ...)
{
  va_list va;

  va_start (va, format);

  warning_va (format, va);

  va_end (va);
}


/// GARBAGE COLLECTOR


struct
{
  struct value *head;
  int allocated;
  int threshold;
} gc = { NULL, 0, 4 };


void
gc_register (struct value *value)
{
  value->next = gc.head;

  gc.head = value;

  gc.allocated++;
}


void
gc_destroy (void)
{
  struct value *p = gc.head;

  while (p)
    {
      struct value *q = p->next;

      value_destroy (p);

      p = q;
    }
}


void
gc_mark (void)
{
  value_mark (vstack_last_pop);

  for (usize i = 0; i < vstack_head; ++i)
    value_mark (vstack[i]);

  for (usize i = 0; i < sstack_head; ++i)
    value_mark (sstack[i]);
}


void
gc_sweep (void)
{
  struct value **p = &gc.head;

  while (*p)
    {
      if (!(*p)->marked)
        {
          struct value *q = *p;

          *p = q->next;

          value_destroy (q);

          gc.allocated--;
        }
      else
        {
          value_unmark (*p);

          p = &(*p)->next;
        }
    }
}


void
gc_collect (void)
{
  gc_mark ();

  gc_sweep ();

  gc.threshold *= 2;
}


void
gc_collect_try (void)
{
  if (gc.allocated > gc.threshold)
    gc_collect ();
}


/// CLOSURE


struct closure *
closure_create (struct value **env, usize env_size, closure_f *f)
{
  struct closure *closure = malloc0 (sizeof (struct closure));

  closure->env = malloc0 (sizeof (struct value *) * env_size);

  for (usize i = 0; i < env_size; ++i)
    closure->env[i] = env[i];

  closure->env_size = env_size;

  closure->f = f;

  return closure;
}


void
closure_destroy (struct closure *closure)
{
  free (closure->env);
  free (closure);
}


void
closure_mark (struct closure *closure)
{
  for (usize i = 0; i < closure->env_size; ++i)
    value_mark (closure->env[i]);
}


/// VALUE


struct value *
value_create (enum value_type t)
{
  struct value *value = malloc0 (sizeof (struct value));

  gc_register (value);

  value->t = t;

  return value;
}


// struct value *
// value_copy (struct value *value)
// {
//   struct value *copy;
// 
//   copy = value_create (value->t);
// 
//   switch (value->t)
//     {
//     case VALUE_FLT:
//       copy->d.flt = value->d.flt;
//       break;
// 
//     case VALUE_BLK:
//       copy->d.blk = value->d.blk;
//       break;
// 
//     default:
//       break;
//     }
// 
//   return copy;
// }


void
value_destroy (struct value *value)
{
  switch (value->t)
    {
    case VALUE_C:
      closure_destroy (value->d.c);
      break;
    default:
      break;
    }

#ifdef DEBUG
  struct value_trace *trace = value->debug_trace;

  while (trace)
    {
      struct value_trace *next = trace->next;

      value_trace_destroy (trace);

      trace = next;
    }
#endif

  free (value);
}


void
value_mark (struct value *value)
{
  if (value->marked)
    return;

  value->marked = true;

  switch (value->t)
    {
    case VALUE_C:
      closure_mark (value->d.c);
      break;
    default:
      break;
    }
}


void
value_unmark (struct value *value)
{
  value->marked = false;
}


struct value *
value_box_u (void)
{
  return value_create (VALUE_U);
}


struct value *
value_box_f (f64 f)
{
  struct value *value = value_create (VALUE_F);

  value->d.f = f;

  return value;
}


struct value *
value_box_b (block_f *b)
{
  struct value *value = value_create (VALUE_B);

  value->d.b = b;

  return value;
}


struct value *
value_box_c (struct value *env[], usize env_size, closure_f *f)
{
  struct value *value = value_create (VALUE_C);

  value->d.c = closure_create (env, env_size, f);

  return value;
}


void
value_unbox_u (struct value *value)
{
#ifdef DEBUG
  panic_assert (value->t == VALUE_U);
#else
  UNUSED (value);
#endif
}


f64
value_unbox_f (struct value *value)
{
#ifdef DEBUG
  panic_assert (value->t == VALUE_F);
#endif

  return value->d.f;
}


block_f *
value_unbox_b (struct value *value)
{
#ifdef DEBUG
  panic_assert (value->t == VALUE_B);
#endif

  return value->d.b;
}


struct closure *
value_unbox_c (struct value *value)
{
#ifdef DEBUG
  panic_assert (value->t == VALUE_C);
#endif

  return value->d.c;
}


static void
print_double (FILE *stream, double x)
{
  char buffer[64];

  sprintf (buffer, "%.12f", x);

  char *p = buffer + strlen (buffer) - 1;

  while (p > buffer && *p == '0')
    p--;

  if (*p == '.')
    p--;

  p[1] = '\0';

  fprintf (stream, "%s", buffer);
}


void
value_fprint (FILE *stream, struct value *value)
{
  switch (value->t)
    {
    case VALUE_U:
      fprintf (stream, "()");
      break;

    case VALUE_F:
      print_double (stream, value->d.f);
      break;

    case VALUE_B:
      fprintf (stream, "(Block)");
      break;

    case VALUE_C:
      fprintf (stream, "(Closure, env=[");

      for (usize i = 0; i < value->d.c->env_size; i++)
        {
          if (i > 0)
            fprintf (stream, ", ");

          value_fprint (stream, value->d.c->env[i]);
        }

      fprintf (stream, "])");
      break;
    }

#ifdef DEBUG
  if (!value->debug_trace)
    return;

  fprintf (stream, " { ");

  for (struct value_trace *trace = value->debug_trace; trace; trace = trace->next)
    {
      // fprintf (stream, "\"%s\": \"%s\"", trace->location, trace->transform);

      fprintf (stream, "\033[92m%s\033[0m %s", trace->transform, trace->location);

      if (trace->next)
        fprintf (stream, " ");
    }

  fprintf (stream, " }");
#endif
}


void
value_fprintn (FILE *stream, struct value *value)
{
  value_fprint (stream, value);

  putc ('\n', stream);
}


void
value_print (struct value *value)
{
  value_fprint (stdout, value);
}


void
value_printn (struct value *value)
{
  value_fprintn (stdout, value);
}


bool
value_bool (struct value *value)
{
  switch (value->t)
    {
    case VALUE_U:
      return true;
    case VALUE_F:
      return value->d.f;
    case VALUE_B:
      return value->d.b;
    case VALUE_C:
      return value->d.c;
    default:
      return false;
    }
}


void
value_execute (struct value *value)
{
#ifdef DEBUG
  panic_assert (value->t == VALUE_B || value->t == VALUE_C);
#endif

  switch (value->t)
    {
    case VALUE_B:
      value->d.b ();
      break;

    case VALUE_C:
      value->d.c->f (value->d.c->env);
      break;

    default:
      break;
    }
}


#ifdef DEBUG
void
value_append_trace (struct value *value, struct value_trace *trace)
{
  value_trace_append (&value->debug_trace, trace);
}
#endif


#endif // RUNTIME_C
